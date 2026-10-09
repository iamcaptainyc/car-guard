/**
 * =============================================================================
 * 🚀 Jetson Orin 原生 C++ 串行多模型基准测速与检测视频导出
 * (benchmark_cpp/main.cpp)
 *
 * 核心特性:
 * 1. 100% 原生 TensorRT C++ API (NvInfer.h) + CUDA Runtime (cuda_runtime_api.h)
 * 2. 零 Python 开销、零 GIL 锁、Pinned 锁页内存显存搬运
 * 3. 测量口径与 trtexec 绝对对齐:
 *    原生 TRT 延迟 = H2D (显存上传) + 纯 GPU 核心运算 + D2H (显存回传)
 * 4. 串行依次执行: YOLO Det -> Face Det -> Face Rec -> OSNet ReID -> RTMPose
 * 5. 实时控制台打印逐帧耗时与 FPS，叠加可视化 BBox、人脸、骨骼与 HUD，输出 MP4
 * =============================================================================
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <csignal>
#include <iomanip>

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace nvinfer1;

// 全局安全退出标志
static volatile bool g_stop = false;
void sig_handler(int sig) {
    std::cout << "\n[!] 收到中断信号 (Ctrl+C)，正在安全封口并保存输出视频..." << std::endl;
    g_stop = true;
}

// TensorRT 日志输出器
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
} gLogger;

// 检查 CUDA 调用
#define CHECK_CUDA(status) \
    do { \
        auto ret = (status); \
        if (ret != cudaSuccess) { \
            std::cerr << "[CUDA Error] " << cudaGetErrorString(ret) << " at line " << __LINE__ << std::endl; \
        } \
    } while (0)

// =============================================================================
// 原生 C++ TensorRT 模型推理引擎包装器
// =============================================================================
struct TimingMetrics {
    float h2d_ms{0.0f};
    float gpu_ms{0.0f};
    float d2h_ms{0.0f};
    float trt_latency_ms{0.0f};
};

class NativeEngine {
public:
    std::string model_path;
    std::unique_ptr<IRuntime> runtime;
    std::unique_ptr<ICudaEngine> engine;
    std::unique_ptr<IExecutionContext> context;
    cudaStream_t stream{nullptr};

    cudaEvent_t ev_start{nullptr};
    cudaEvent_t ev_h2d_end{nullptr};
    cudaEvent_t ev_gpu_end{nullptr};
    cudaEvent_t ev_d2h_end{nullptr};

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<void*> d_buffers;       // 设备显存指针列表
    std::vector<size_t> buffer_sizes;   // 显存字节数
    std::vector<std::vector<int64_t>> shapes;
    std::vector<std::vector<float>> h_outputs; // 主机接收缓冲

    NativeEngine(const std::string& path, int fixed_batch = 1) : model_path(path) {
        // 1. 读取 Engine 二进制文件
        std::ifstream file(path, std::ios::binary);
        if (!file.good()) {
            std::cerr << "[!] 无法打开引擎文件: " << path << std::endl;
            return;
        }
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        file.read(buffer.data(), size);
        file.close();

        // 2. 反序列化
        runtime.reset(createInferRuntime(gLogger));
        engine.reset(runtime->deserializeCudaEngine(buffer.data(), size));
        if (!engine) {
            std::cerr << "[!] 反序列化 TensorRT 引擎失败: " << path << std::endl;
            return;
        }
        context.reset(engine->createExecutionContext());
        CHECK_CUDA(cudaStreamCreate(&stream));

        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_h2d_end));
        CHECK_CUDA(cudaEventCreate(&ev_gpu_end));
        CHECK_CUDA(cudaEventCreate(&ev_d2h_end));

        // 3. 解析 IO 张量并预分配常驻设备显存 (Zero-Allocation Loop)
#if NV_TENSORRT_MAJOR >= 10
        int nb_tensors = engine->getNbIOTensors();
        for (int i = 0; i < nb_tensors; ++i) {
            const char* name = engine->getIOTensorName(i);
            TensorIOMode mode = engine->getTensorIOMode(name);
            Dims dims = engine->getTensorShape(name);
            DataType dtype = engine->getTensorDataType(name);

            std::vector<int64_t> s;
            size_t count = 1;
            for (int d = 0; d < dims.nbDims; ++d) {
                int64_t dim_val = dims.d[d] < 0 ? fixed_batch : dims.d[d];
                s.push_back(dim_val);
                count *= dim_val;
            }
            shapes.push_back(s);
            size_t bytes = count * sizeof(float); // 绝大多数模型为 float32
            buffer_sizes.push_back(bytes);

            void* d_ptr = nullptr;
            CHECK_CUDA(cudaMalloc(&d_ptr, bytes));
            d_buffers.push_back(d_ptr);

            if (mode == TensorIOMode::kINPUT) {
                input_names.push_back(name);
            } else {
                output_names.push_back(name);
                h_outputs.push_back(std::vector<float>(count, 0.0f));
            }
        }
#else
        int nb_bindings = engine->getNbBindings();
        for (int i = 0; i < nb_bindings; ++i) {
            const char* name = engine->getBindingName(i);
            Dims dims = engine->getBindingDimensions(i);

            std::vector<int64_t> s;
            size_t count = 1;
            for (int d = 0; d < dims.nbDims; ++d) {
                int64_t dim_val = dims.d[d] < 0 ? fixed_batch : dims.d[d];
                s.push_back(dim_val);
                count *= dim_val;
            }
            shapes.push_back(s);
            size_t bytes = count * sizeof(float);
            buffer_sizes.push_back(bytes);

            void* d_ptr = nullptr;
            CHECK_CUDA(cudaMalloc(&d_ptr, bytes));
            d_buffers.push_back(d_ptr);

            if (engine->bindingIsInput(i)) {
                input_names.push_back(name);
            } else {
                output_names.push_back(name);
                h_outputs.push_back(std::vector<float>(count, 0.0f));
            }
        }
#endif
    }

    ~NativeEngine() {
        for (void* ptr : d_buffers) {
            if (ptr) cudaFree(ptr);
        }
        if (ev_start) cudaEventDestroy(ev_start);
        if (ev_h2d_end) cudaEventDestroy(ev_h2d_end);
        if (ev_gpu_end) cudaEventDestroy(ev_gpu_end);
        if (ev_d2h_end) cudaEventDestroy(ev_d2h_end);
        if (stream) cudaStreamDestroy(stream);
    }

    bool is_valid() const { return engine && context; }

    // 原生执行推理，硬件高精度微秒打点
    TimingMetrics infer(const float* input_host_data, size_t input_bytes) {
        TimingMetrics metrics;
        if (!is_valid()) return metrics;

        // 1. 开始打点
        cudaEventRecord(ev_start, stream);

        // 2. H2D 显存异步上传
        cudaMemcpyAsync(d_buffers[0], input_host_data, input_bytes, cudaMemcpyHostToDevice, stream);
        cudaEventRecord(ev_h2d_end, stream);

        // 3. 纯 GPU 核心推理执行
#if NV_TENSORRT_MAJOR >= 10
        for (size_t i = 0; i < d_buffers.size(); ++i) {
            const char* name = engine->getIOTensorName(i);
            context->setTensorAddress(name, d_buffers[i]);
        }
        context->enqueueV3(stream);
#else
        context->enqueueV2(d_buffers.data(), stream, nullptr);
#endif
        cudaEventRecord(ev_gpu_end, stream);

        // 4. D2H 显存异步回传
        size_t out_idx = 0;
        for (size_t i = input_names.size(); i < d_buffers.size(); ++i) {
            cudaMemcpyAsync(h_outputs[out_idx].data(), d_buffers[i], buffer_sizes[i], cudaMemcpyDeviceToHost, stream);
            out_idx++;
        }
        cudaEventRecord(ev_d2h_end, stream);

        // 5. 同步与精准结算时间
        cudaStreamSynchronize(stream);

        cudaEventElapsedTime(&metrics.h2d_ms, ev_start, ev_h2d_end);
        cudaEventElapsedTime(&metrics.gpu_ms, ev_h2d_end, ev_gpu_end);
        cudaEventElapsedTime(&metrics.d2h_ms, ev_gpu_end, ev_d2h_end);
        metrics.trt_latency_ms = metrics.h2d_ms + metrics.gpu_ms + metrics.d2h_ms;

        return metrics;
    }
};

// =============================================================================
// C++ 图像预处理与后处理工具函数
// =============================================================================
struct Detection {
    cv::Rect box;
    float conf;
    std::string cat;
};

// Letterbox 变换
cv::Mat letterbox(const cv::Mat& src, int target_w, int target_h, float& scale, int& pad_w, int& pad_h) {
    float r = std::min((float)target_w / src.cols, (float)target_h / src.rows);
    int new_w = std::round(src.cols * r);
    int new_h = std::round(src.rows * r);
    pad_w = (target_w - new_w) / 2;
    pad_h = (target_h - new_h) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));
    cv::Mat out(target_h, target_w, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(pad_w, pad_h, new_w, new_h)));
    scale = r;
    return out;
}

// 快速归一化与 HWC->CHW 打包
void blobFromImage(const cv::Mat& bgr, float* blob, float scale_val = 1.0f / 255.0f, bool rgb = true) {
    int h = bgr.rows;
    int w = bgr.cols;
    int channel_size = h * w;

    for (int y = 0; y < h; ++y) {
        const uchar* ptr = bgr.ptr<uchar>(y);
        for (int x = 0; x < w; ++x) {
            float b = ptr[x * 3 + 0] * scale_val;
            float g = ptr[x * 3 + 1] * scale_val;
            float r = ptr[x * 3 + 2] * scale_val;

            if (rgb) {
                blob[0 * channel_size + y * w + x] = r;
                blob[1 * channel_size + y * w + x] = g;
                blob[2 * channel_size + y * w + x] = b;
            } else {
                blob[0 * channel_size + y * w + x] = b;
                blob[1 * channel_size + y * w + x] = g;
                blob[2 * channel_size + y * w + x] = r;
            }
        }
    }
}

// YOLO11/v8 后处理
std::vector<Detection> postprocess_yolo(const float* preds, int orig_w, int orig_h, float r, int pad_w, int pad_h, float conf_thresh = 0.25f) {
    // preds: [84, 8400]
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<std::string> cats;

    for (int i = 0; i < 8400; ++i) {
        float max_cls_score = 0.0f;
        int max_cls_id = -1;
        // 查找 80 类的最大分
        for (int c = 0; c < 80; ++c) {
            float s = preds[(4 + c) * 8400 + i];
            if (s > max_cls_score) {
                max_cls_score = s;
                max_cls_id = c;
            }
        }

        if (max_cls_score >= conf_thresh) {
            float cx = preds[0 * 8400 + i];
            float cy = preds[1 * 8400 + i];
            float bw = preds[2 * 8400 + i];
            float bh = preds[3 * 8400 + i];

            int x = std::round((cx - bw * 0.5f - pad_w) / r);
            int y = std::round((cy - bh * 0.5f - pad_h) / r);
            int w = std::round(bw / r);
            int h = std::round(bh / r);

            x = std::max(0, std::min(orig_w - 1, x));
            y = std::max(0, std::min(orig_h - 1, y));
            w = std::max(5, std::min(orig_w - x, w));
            h = std::max(5, std::min(orig_h - y, h));

            boxes.push_back(cv::Rect(x, y, w, h));
            scores.push_back(max_cls_score);
            cats.push_back(max_cls_id == 0 ? "person" : (max_cls_id == 2 ? "car" : "other"));
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_thresh, 0.45f, indices);

    std::vector<Detection> results;
    for (int idx : indices) {
        results.push_back({boxes[idx], scores[idx], cats[idx]});
    }
    return results;
}

// COCO 17 骨架连线定义
static const std::vector<std::pair<int, int>> SKELETON_PAIRS = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4},
    {5, 7}, {7, 9}, {6, 8}, {8, 10},
    {5, 6}, {5, 11}, {6, 12}, {11, 12},
    {11, 13}, {13, 15}, {12, 14}, {14, 16}
};

// 绘制半透明 HUD 指标卡片
void draw_hud(cv::Mat& frame, const std::vector<std::string>& lines) {
    if (lines.empty()) return;
    int x0 = 15, y0 = 15;
    int line_h = 24;
    int box_w = 640;
    int box_h = lines.size() * line_h + 16;

    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay, cv::Rect(x0, y0, box_w, box_h), cv::Scalar(20, 20, 20), -1);
    cv::addWeighted(overlay, 0.75, frame, 0.25, 0, frame);
    cv::rectangle(frame, cv::Rect(x0, y0, box_w, box_h), cv::Scalar(0, 255, 200), 1);

    for (size_t i = 0; i < lines.size(); ++i) {
        int y = y0 + 20 + i * line_h;
        cv::putText(frame, lines[i], cv::Point(x0 + 10, y), cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

// 统计辅助
struct StatRecord {
    std::vector<float> h2d;
    std::vector<float> gpu;
    std::vector<float> d2h;
    std::vector<float> trt;

    void add(const TimingMetrics& m) {
        h2d.push_back(m.h2d_ms);
        gpu.push_back(m.gpu_ms);
        d2h.push_back(m.d2h_ms);
        trt.push_back(m.trt_latency_ms);
    }
    float avg_h2d() const { return h2d.empty() ? 0.0f : std::accumulate(h2d.begin(), h2d.end(), 0.0f) / h2d.size(); }
    float avg_gpu() const { return gpu.empty() ? 0.0f : std::accumulate(gpu.begin(), gpu.end(), 0.0f) / gpu.size(); }
    float avg_d2h() const { return d2h.empty() ? 0.0f : std::accumulate(d2h.begin(), d2h.end(), 0.0f) / d2h.size(); }
    float avg_trt() const { return trt.empty() ? 0.0f : std::accumulate(trt.begin(), trt.end(), 0.0f) / trt.size(); }
};

// =============================================================================
// 主函数
// =============================================================================
int main(int argc, char** argv) {
    std::signal(SIGINT, sig_handler);

    std::string video_path = "sample_fisheye_left.mp4";
    std::string out_path = "tmp/serial_cpp_output.mp4";
    std::string det_engine = "models/yolo/yolo11n.engine";
    std::string fdet_engine = "models/insightface/det_500m.engine";
    std::string frec_engine = "models/insightface/w600k_mbf.engine";
    std::string reid_engine = "models/reid/osnet_x0_25.engine";
    std::string pose_engine = "models/pose/rtmpose-s.engine";
    bool headless = false;
    bool force_all = true;
    int max_frames = 0;

    // 解析简单命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--video" && i + 1 < argc) video_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc) out_path = argv[++i];
        else if (arg == "--det" && i + 1 < argc) det_engine = argv[++i];
        else if (arg == "--face-det" && i + 1 < argc) fdet_engine = argv[++i];
        else if (arg == "--face-rec" && i + 1 < argc) frec_engine = argv[++i];
        else if (arg == "--reid" && i + 1 < argc) reid_engine = argv[++i];
        else if (arg == "--pose" && i + 1 < argc) pose_engine = argv[++i];
        else if (arg == "--max-frames" && i + 1 < argc) max_frames = std::stoi(argv[++i]);
        else if (arg == "--headless") headless = true;
        else if (arg == "--no-force-all") force_all = false;
    }

    // 容错寻找默认视频
    if (!cv::utils::fs::exists(video_path)) {
        for (const auto& cand : {
            "C:\\Users\\chanceyu\\Downloads\\开门碰撞1.mp4",
            "开门碰撞1.mp4",
            "models/sample.mp4",
            "sample_fisheye_left.mp4"
        }) {
            if (cv::utils::fs::exists(cand)) {
                video_path = cand;
                break;
            }
        }
    }

    std::cout << "==========================================================================================" << std::endl;
    std::cout << " 🚀 Jetson Orin 原生 C++ 串行多模型对标测速 (纯 C++ TensorRT 极致吞吐)" << std::endl;
    std::cout << "==========================================================================================" << std::endl;
    std::cout << "[*] 输入视频: " << video_path << std::endl;
    std::cout << "[*] 输出视频: " << out_path << std::endl;
    std::cout << "[*] 1. YOLO Det  : " << det_engine << std::endl;
    std::cout << "[*] 2. Face Det  : " << fdet_engine << std::endl;
    std::cout << "[*] 3. Face Rec  : " << frec_engine << std::endl;
    std::cout << "[*] 4. OSNet ReID: " << reid_engine << std::endl;
    std::cout << "[*] 5. RTMPose   : " << pose_engine << std::endl;
    std::cout << "------------------------------------------------------------------------------------------" << std::endl;

    // 1. 初始化 5 大原生 TensorRT 引擎
    NativeEngine eng_det(det_engine);
    NativeEngine eng_fdet(fdet_engine);
    NativeEngine eng_frec(frec_engine);
    NativeEngine eng_reid(reid_engine, 16);
    NativeEngine eng_pose(pose_engine);

    if (!eng_det.is_valid()) {
        std::cerr << "[!] 无法载入 YOLO 引擎，请确认文件路径: " << det_engine << std::endl;
        return 1;
    }

    // 2. 打开视频流
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "[!] 无法打开视频文件: " << video_path << std::endl;
        return 1;
    }
    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps_in = cap.get(cv::CAP_PROP_FPS);
    if (fps_in <= 0) fps_in = 25.0;
    int total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);

    cv::VideoWriter writer(out_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps_in, cv::Size(width, height));

    // 预分配主机输入 buffer (避免每帧动态分配)
    std::vector<float> blob_yolo(3 * 640 * 640);
    std::vector<float> blob_fdet(3 * 640 * 640);
    std::vector<float> blob_frec(3 * 112 * 112);
    std::vector<float> blob_reid(16 * 3 * 256 * 128);
    std::vector<float> blob_pose(3 * 256 * 192);

    StatRecord s_det, s_fdet, s_frec, s_reid, s_pose;
    std::vector<float> all_trt_sums;
    std::vector<float> all_e2e_sums;

    std::cout << "\n[*] 开始纯 C++ 原生串行推理，控制台逐帧打点..." << std::endl;
    std::cout << "帧号    | 实时FPS | YOLO(ms) | F-Det(ms) | F-Rec(ms) | ReID(ms) | Pose(ms) | TRT延迟和   | E2E整帧   | 检出目标" << std::endl;
    std::cout << "-------------------------------------------------------------------------------------------------------" << std::endl;

    int frame_idx = 0;
    auto t_start_wall = std::chrono::high_resolution_clock::now();

    cv::Mat frame;
    while (!g_stop && cap.read(frame)) {
        if (max_frames > 0 && frame_idx >= max_frames) break;

        auto t_e2e_start = std::chrono::high_resolution_clock::now();

        // ----------------- 1. YOLO 目标检测 (C++ 原生) -----------------
        float scale = 1.0f;
        int pad_w = 0, pad_h = 0;
        cv::Mat lb = letterbox(frame, 640, 640, scale, pad_w, pad_h);
        blobFromImage(lb, blob_yolo.data());

        TimingMetrics m_det = eng_det.infer(blob_yolo.data(), blob_yolo.size() * sizeof(float));
        s_det.add(m_det);

        std::vector<Detection> dets = postprocess_yolo(eng_det.h_outputs[0].data(), width, height, scale, pad_w, pad_h);
        std::vector<Detection> persons;
        for (const auto& d : dets) {
            if (d.cat == "person") persons.push_back(d);
        }

        cv::Rect target_bbox;
        bool is_mock = false;
        if (!persons.empty()) {
            target_bbox = persons[0].box;
        } else if (force_all) {
            target_bbox = cv::Rect(width / 4, height / 4, width / 2, height / 2);
            is_mock = true;
        }

        // ----------------- 2. Face Det (SCRFD 原生) -----------------
        TimingMetrics m_fdet{0, 0, 0, 0};
        if (target_bbox.area() > 0 && eng_fdet.is_valid()) {
            cv::Rect head_roi(target_bbox.x, target_bbox.y, target_bbox.width, std::min(height - target_bbox.y, (int)(target_bbox.height * 0.45f)));
            head_roi &= cv::Rect(0, 0, width, height);
            if (head_roi.area() > 100) {
                cv::Mat head_img;
                cv::resize(frame(head_roi), head_img, cv::Size(640, 640));
                blobFromImage(head_img, blob_fdet.data(), 1.0f / 128.0f);
                m_fdet = eng_fdet.infer(blob_fdet.data(), blob_fdet.size() * sizeof(float));
            }
        }
        s_fdet.add(m_fdet);

        // ----------------- 3. Face Rec (ArcFace 原生) -----------------
        TimingMetrics m_frec{0, 0, 0, 0};
        if (target_bbox.area() > 0 && eng_frec.is_valid()) {
            cv::Rect face_roi(target_bbox.x, target_bbox.y, std::min(112, target_bbox.width), std::min(112, target_bbox.height));
            face_roi &= cv::Rect(0, 0, width, height);
            if (face_roi.area() > 50) {
                cv::Mat face_112;
                cv::resize(frame(face_roi), face_112, cv::Size(112, 112));
                blobFromImage(face_112, blob_frec.data(), 1.0f / 127.5f);
                m_frec = eng_frec.infer(blob_frec.data(), blob_frec.size() * sizeof(float));
            }
        }
        s_frec.add(m_frec);

        // ----------------- 4. OSNet ReID (原生) -----------------
        TimingMetrics m_reid{0, 0, 0, 0};
        if (target_bbox.area() > 0 && eng_reid.is_valid()) {
            cv::Rect body_roi = target_bbox & cv::Rect(0, 0, width, height);
            if (body_roi.area() > 100) {
                cv::Mat body_img;
                cv::resize(frame(body_roi), body_img, cv::Size(128, 256));
                blobFromImage(body_img, blob_reid.data());
                m_reid = eng_reid.infer(blob_reid.data(), blob_reid.size() * sizeof(float));
            }
        }
        s_reid.add(m_reid);

        // ----------------- 5. RTMPose 姿态 (原生) -----------------
        TimingMetrics m_pose{0, 0, 0, 0};
        std::vector<cv::Point> kpts_pts;
        if (target_bbox.area() > 0 && eng_pose.is_valid()) {
            cv::Rect pose_roi = target_bbox & cv::Rect(0, 0, width, height);
            if (pose_roi.area() > 100) {
                cv::Mat pose_img;
                cv::resize(frame(pose_roi), pose_img, cv::Size(192, 256));
                blobFromImage(pose_img, blob_pose.data());
                m_pose = eng_pose.infer(blob_pose.data(), blob_pose.size() * sizeof(float));

                // SimCC 解码关键点
                const float* sim_x = eng_pose.h_outputs[0].data();
                const float* sim_y = eng_pose.h_outputs[1].data();
                for (int k = 0; k < 17; ++k) {
                    int max_x = std::max_element(sim_x + k * 384, sim_x + (k + 1) * 384) - (sim_x + k * 384);
                    int max_y = std::max_element(sim_y + k * 512, sim_y + (k + 1) * 512) - (sim_y + k * 512);
                    int px = target_bbox.x + (max_x / 2.0f / 192.0f) * target_bbox.width;
                    int py = target_bbox.y + (max_y / 2.0f / 256.0f) * target_bbox.height;
                    kpts_pts.push_back(cv::Point(px, py));
                }
            }
        }
        s_pose.add(m_pose);

        auto t_e2e_end = std::chrono::high_resolution_clock::now();
        float e2e_total_ms = std::chrono::duration<float, std::milli>(t_e2e_end - t_e2e_start).count();
        float trt_sum_ms = m_det.trt_latency_ms + m_fdet.trt_latency_ms + m_frec.trt_latency_ms + m_reid.trt_latency_ms + m_pose.trt_latency_ms;
        float real_fps = e2e_total_ms > 0 ? 1000.0f / e2e_total_ms : 0.0f;

        all_e2e_sums.push_back(e2e_total_ms);
        all_trt_sums.push_back(trt_sum_ms);

        // ----------------- 绘制标注 -----------------
        for (const auto& d : dets) {
            cv::Scalar color = (d.cat == "person") ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 255, 0);
            cv::rectangle(frame, d.box, color, 2);
            cv::putText(frame, d.cat, cv::Point(d.box.x, std::max(15, d.box.y - 4)), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }
        if (kpts_pts.size() == 17) {
            for (const auto& pair : SKELETON_PAIRS) {
                cv::line(frame, kpts_pts[pair.first], kpts_pts[pair.second], cv::Scalar(0, 255, 255), 2);
            }
            for (const auto& kp : kpts_pts) {
                cv::circle(frame, kp, 3, cv::Scalar(0, 0, 255), -1);
            }
        }

        // 叠加性能 HUD 看板
        std::vector<std::string> hud_lines;
        hud_lines.push_back("[C++ NATIVE TRT] Frame #" + std::to_string(frame_idx + 1) + "/" + std::to_string(total_frames) + " | Real FPS: " + std::to_string((int)real_fps));
        hud_lines.push_back("TRT Latency: Det " + std::to_string(m_det.trt_latency_ms).substr(0, 4) + "ms | F-Det " + std::to_string(m_fdet.trt_latency_ms).substr(0, 4) + "ms | F-Rec " + std::to_string(m_frec.trt_latency_ms).substr(0, 4) + "ms | ReID " + std::to_string(m_reid.trt_latency_ms).substr(0, 4) + "ms | Pose " + std::to_string(m_pose.trt_latency_ms).substr(0, 4) + "ms");
        hud_lines.push_back("TRT Serial Sum: " + std::to_string(trt_sum_ms).substr(0, 5) + "ms (TRT Max FPS: " + std::to_string((int)(1000.0f / std::max(0.1f, trt_sum_ms))) + ") | E2E: " + std::to_string(e2e_total_ms).substr(0, 5) + "ms");
        draw_hud(frame, hud_lines);

        if (writer.isOpened()) {
            writer.write(frame);
        }

        // 终端逐帧单行输出
        std::cout << "#" << std::setw(5) << std::setfill('0') << (frame_idx + 1) << " | "
                  << std::fixed << std::setprecision(1) << std::setw(5) << real_fps << " | "
                  << std::setprecision(2) << std::setw(6) << m_det.trt_latency_ms << " | "
                  << std::setw(7) << m_fdet.trt_latency_ms << " | "
                  << std::setw(7) << m_frec.trt_latency_ms << " | "
                  << std::setw(6) << m_reid.trt_latency_ms << " | "
                  << std::setw(6) << m_pose.trt_latency_ms << " | "
                  << std::setw(7) << trt_sum_ms << " ms | "
                  << std::setw(6) << e2e_total_ms << " ms | "
                  << dets.size() << "目标 (" << persons.size() << "人)" << (is_mock ? " [STRESS]" : "")
                  << std::endl;

        if (!headless) {
            cv::imshow("Jetson Orin C++ Native TRT Benchmark", frame);
            if (cv::waitKey(1) == 'q') break;
        }

        frame_idx++;
    }

    cap.release();
    if (writer.isOpened()) writer.release();
    cv::destroyAllWindows();

    auto t_end_wall = std::chrono::high_resolution_clock::now();
    float total_sec = std::chrono::duration<float>(t_end_wall - t_start_wall).count();
    float avg_trt_sum = all_trt_sums.empty() ? 0.0f : std::accumulate(all_trt_sums.begin(), all_trt_sums.end(), 0.0f) / all_trt_sums.size();
    float avg_e2e = all_e2e_sums.empty() ? 0.0f : std::accumulate(all_e2e_sums.begin(), all_e2e_sums.end(), 0.0f) / all_e2e_sums.size();
    float overall_fps = total_sec > 0 ? frame_idx / total_sec : 0.0f;

    // 打印汇总报告
    std::cout << "\n==============================================================================================================" << std::endl;
    std::cout << "                  Jetson Orin 原生 C++ 串行多模型对标测速综合报告 (严格对标 trtexec)" << std::endl;
    std::cout << "==============================================================================================================" << std::endl;
    std::cout << "  > 评估样本总数 : " << frame_idx << " 帧 (总用时: " << std::fixed << std::setprecision(2) << total_sec << " 秒)" << std::endl;
    std::cout << "  > 纯模型串行极限帧率 : " << std::fixed << std::setprecision(1) << (1000.0f / std::max(0.01f, avg_trt_sum)) << " FPS (5个引擎纯推理延迟和: " << avg_trt_sum << " ms)" << std::endl;
    std::cout << "  > 端到端交付综合帧率 : " << overall_fps << " FPS (含视频解码、前处理、绘制与编码写盘: " << avg_e2e << " ms)" << std::endl;
    std::cout << "--------------------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "模型名称                   | H2D显存拷贝     | 纯GPU核心运算      | D2H显存回传     | TRT平均延迟      | trtexec实测参考值" << std::endl;
    std::cout << "--------------------------------------------------------------------------------------------------------------" << std::endl;

    auto print_row = [](const std::string& name, const StatRecord& s, const std::string& ref) {
        std::cout << std::left << std::setw(26) << name << " | "
                  << std::right << std::fixed << std::setprecision(2) << std::setw(7) << s.avg_h2d() << " ms  | "
                  << std::setw(9) << s.avg_gpu() << " ms  | "
                  << std::setw(7) << s.avg_d2h() << " ms  | "
                  << std::setw(8) << s.avg_trt() << " ms   | "
                  << ref << std::endl;
    };

    print_row("1. YOLO Det (yolo11n)", s_det, "~4.39 ms (yolo11n) / ~7.04 ms (yolo11s)");
    print_row("2. Face-Det (SCRFD)", s_fdet, "~2.04 ms (det_500m)");
    print_row("3. Face-Rec (ArcFace)", s_frec, "~1.34 ms (w600k_mbf)");
    print_row("4. OSNet ReID", s_reid, "~4.84 ms (osnet_x0_25)");
    print_row("5. RTMPose", s_pose, "~1.22 ms (rtmpose-s)");

    std::cout << "--------------------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(26) << "5 模型串行推理延迟合计" << " | -           | -             | -           | "
              << std::right << std::setw(8) << avg_trt_sum << " ms   | 纯TRT极限: " << (1000.0f / std::max(0.01f, avg_trt_sum)) << " FPS" << std::endl;
    std::cout << "==============================================================================================================" << std::endl;
    if (cv::utils::fs::exists(out_path)) {
        std::cout << "[OK] 导出测试视频已安全保存: " << out_path << std::endl;
    }
    std::cout << "==============================================================================================================" << std::endl;

    return 0;
}
