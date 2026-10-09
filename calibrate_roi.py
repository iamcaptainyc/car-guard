"""
智能汽车露营守卫 - 车身与敏感部件固定 ROI 可视化标定工具 (calibrate_roi.py)

功能：
1. 针对车载环视鱼眼相机（前/左/右/后）图像或视频帧，以鼠标交互方式多边形绘制固定监控 ROI；
2. 支持“敏感部件防盗类”（门把手、充电口、后备箱锁）与“车身防撞类”（车侧门板、后保险杠等）；
3. 自动几何合法性校验（越界检测、最小面积限制、顶点闭合）；
4. 标定结果一键同步持久化至 roi_config.json 和 vehicle_calib.json。

使用示例：
  python calibrate_roi.py --image sample_fisheye.png --camera-id 1
  python calibrate_roi.py --video sample_fisheye_left.mp4 --camera-id 1
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np

from camping_guard.types import Zone3ROI
from camping_guard.tools.roi_calibrator import ROICalibrator


# 常用预置待标定部件定义 (方便按数字键快速切换)
PRESET_TEMPLATES = {
    1: {"roi_id": "handle_fl", "name": "左前门把手", "camera_id": 1, "threshold_ratio": 0.10, "roi_type": "part"},
    2: {"roi_id": "handle_rl", "name": "左后门把手", "camera_id": 1, "threshold_ratio": 0.10, "roi_type": "part"},
    3: {"roi_id": "charge_port", "name": "快慢充电口", "camera_id": 1, "threshold_ratio": 0.12, "roi_type": "part"},
    4: {"roi_id": "body_flank_l", "name": "左侧车身门板防碰撞区", "camera_id": 1, "threshold_ratio": 0.08, "roi_type": "body"},
    5: {"roi_id": "body_flank_r", "name": "右侧车身门板防碰撞区", "camera_id": 2, "threshold_ratio": 0.08, "roi_type": "body"},
    6: {"roi_id": "trunk_lock", "name": "后备箱开关与锁扣", "camera_id": 3, "threshold_ratio": 0.12, "roi_type": "part"},
    7: {"roi_id": "body_rear_bumper", "name": "后保险杠与尾门防撞区", "camera_id": 3, "threshold_ratio": 0.08, "roi_type": "body"},
    8: {"roi_id": "roof_rack", "name": "车顶行李架", "camera_id": 3, "threshold_ratio": 0.15, "roi_type": "part"},
}


class InteractiveCalibrateApp:
    def __init__(
        self,
        base_frame: np.ndarray,
        camera_id: int = 1,
        config_path: str = "roi_config.json",
        vehicle_calib_path: str = "vehicle_calib.json",
        sync_vehicle: bool = True,
        cap: Optional[cv2.VideoCapture] = None
    ):
        self.frame = base_frame.copy()
        self.cap = cap
        self.camera_id = camera_id
        self.config_path = config_path
        self.vehicle_calib_path = vehicle_calib_path
        self.sync_vehicle = sync_vehicle

        self.calibrator = ROICalibrator(config_path)
        self.all_rois: List[Zone3ROI] = self.calibrator.load_rois()

        # 当前正在绘制的多边形顶点
        self.current_pts: List[Tuple[int, int]] = []
        # 当前选中的待标定预置模板
        self.current_template = PRESET_TEMPLATES.get(1).copy()
        self.current_template["camera_id"] = self.camera_id

        self.window_name = f"Camping Guard - ROI Calibrator (Camera {self.camera_id})"
        self.status_msg = "就绪: 鼠标左键点击画面绘制多边形顶点"
        self.status_color = (0, 255, 0)
        self.is_video_paused = True

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_pts.append((x, y))
            self.status_msg = f"已添加顶点 ({x}, {y})，当前共 {len(self.current_pts)} 个点"
            self.status_color = (0, 255, 255)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.current_pts:
                removed = self.current_pts.pop()
                self.status_msg = f"已撤销顶点 {removed}，剩余 {len(self.current_pts)} 个点"
                self.status_color = (0, 200, 255)

    def draw_ui(self) -> np.ndarray:
        display = self.frame.copy()
        h, w = display.shape[:2]

        # 1. 渲染当前相机已保存的 ROI
        overlay = display.copy()
        for roi in self.all_rois:
            if roi.camera_id != self.camera_id or len(roi.polygon) < 3:
                continue
            pts = np.array(roi.polygon, dtype=np.int32)
            is_body = getattr(roi, "roi_type", "part") == "body"
            # 部件类用黄色，车身防撞类用青/橙色
            fill_color = (0, 140, 255) if is_body else (0, 220, 220)
            line_color = (0, 100, 255) if is_body else (0, 255, 255)

            cv2.fillPoly(overlay, [pts], fill_color)
            cv2.polylines(display, [pts], True, line_color, 2)

            # 绘制中心标签
            moments = cv2.moments(pts)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx, cy = roi.polygon[0]

            tag = f"{roi.name} ({roi.roi_id})"
            cv2.putText(display, tag, (max(10, cx - 40), max(20, cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(display, tag, (max(10, cx - 40), max(20, cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # 融合半透明已存图层
        cv2.addWeighted(overlay, 0.28, display, 0.72, 0, display)

        # 2. 渲染当前正在绘制的多边形
        if len(self.current_pts) > 0:
            for i, p in enumerate(self.current_pts):
                cv2.circle(display, p, 5, (0, 0, 255), -1)
                cv2.putText(display, str(i + 1), (p[0] + 6, p[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            if len(self.current_pts) >= 2:
                cv2.polylines(display, [np.array(self.current_pts, np.int32)], False, (0, 255, 0), 2)

            if len(self.current_pts) >= 3:
                curr_overlay = display.copy()
                cv2.fillPoly(curr_overlay, [np.array(self.current_pts, np.int32)], (0, 255, 100))
                cv2.addWeighted(curr_overlay, 0.35, display, 0.65, 0, display)

        # 3. 顶部 HUD 仪表盘与操作指南
        hud = display.copy()
        cv2.rectangle(hud, (0, 0), (w, 85), (20, 20, 20), -1)
        cv2.addWeighted(hud, 0.85, display, 0.15, 0, display)

        # 当前标定对象
        target_str = f"当前标定目标: [{self.current_template['name']}] (ID: {self.current_template['roi_id']}, 类型: {self.current_template.get('roi_type', 'part')})"
        cv2.putText(display, target_str, (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # 快捷键指引
        keys_str = "[1~8]:切换预置部件 | [s]:保存当前多边形 | [r]:重置重画 | [d]:删除该部件 | [w]:一键落盘 | [q]:退出"
        cv2.putText(display, keys_str, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        # 底部状态通知栏
        status_hud = display.copy()
        cv2.rectangle(status_hud, (0, h - 30), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(status_hud, 0.85, display, 0.15, 0, display)
        cv2.putText(display, f"状态: {self.status_msg}", (15, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, self.status_color, 1)

        return display

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print("\n" + "=" * 70)
        print(f" 智能汽车露营守卫 - ROI 交互式可视化标定工具已启动 (Camera {self.camera_id})")
        print("=" * 70)
        print(" [鼠标操作]:")
        print("   * 鼠标左键点击  : 添加多边形顶点")
        print("   * 鼠标右键点击  : 撤销上一个顶点")
        print(" [键盘快捷键]:")
        print("   * [1~8] 数字键  : 快速切换预置标定部件 (1:左前把手 2:左后把手 3:充电口 4:左侧防撞区...)")
        print("   * [s] 键        : 确认并暂存当前绘制的多边形 ROI")
        print("   * [r] 键        : 清空重置当前正在画的多边形")
        print("   * [d] 键        : 删除当前预置部件已有的标定结果")
        print("   * [w] 键        : 一键将所有标定结果写入 JSON 文件")
        print("   * [SPACE] 空格  : (视频源模式) 暂停/播放以选择最清晰平稳的车辆帧")
        print("   * [q / ESC]     : 保存并退出")
        print("=" * 70 + "\n")

        while True:
            # 视频源模式支持空格播放/暂停选帧
            if self.cap is not None and not self.is_video_paused:
                ret, next_frame = self.cap.read()
                if ret:
                    self.frame = next_frame
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            display = self.draw_ui()
            cv2.imshow(self.window_name, display)

            k = cv2.waitKey(30) & 0xFF
            if k in [ord('q'), 27]:
                print("[*] 正在退出标定工具...")
                break

            elif k == ord(' '):
                if self.cap is not None:
                    self.is_video_paused = not self.is_video_paused
                    self.status_msg = "已暂停选帧" if self.is_video_paused else "视频播放中..."
                    self.status_color = (255, 200, 0)

            # 数字键切换模板
            elif ord('1') <= k <= ord('8'):
                idx = k - ord('0')
                if idx in PRESET_TEMPLATES:
                    self.current_template = PRESET_TEMPLATES[idx].copy()
                    self.current_template["camera_id"] = self.camera_id
                    self.current_pts.clear()
                    self.status_msg = f"已切换至标定目标: 【{self.current_template['name']}】({self.current_template['roi_id']})"
                    self.status_color = (0, 255, 255)
                    print(f"[*] 切换目标: {self.current_template['name']}")

            # 保存当前多边形
            elif k == ord('s'):
                if len(self.current_pts) < 3:
                    self.status_msg = "错误: 多边形顶点数不足 (< 3)，无法保存！"
                    self.status_color = (0, 0, 255)
                    print("[!] 顶点不足 3 个，请继续点击添加")
                    continue

                new_roi = Zone3ROI(
                    roi_id=self.current_template["roi_id"],
                    name=self.current_template["name"],
                    camera_id=self.camera_id,
                    polygon=self.current_pts.copy(),
                    threshold_ratio=self.current_template.get("threshold_ratio", 0.10),
                    confirm_frames=2,
                    roi_type=self.current_template.get("roi_type", "part")
                )

                is_valid, err_msg = self.calibrator.validate_roi(new_roi, self.frame.shape[:2])
                if not is_valid:
                    self.status_msg = f"校验失败: {err_msg}"
                    self.status_color = (0, 0, 255)
                    print(f"[!] 几何校验未通过: {err_msg}")
                    continue

                # 替换或追加
                self.all_rois = [r for r in self.all_rois if not (r.roi_id == new_roi.roi_id and r.camera_id == new_roi.camera_id)]
                self.all_rois.append(new_roi)

                self.status_msg = f"【成功暂存】{new_roi.name} ({len(new_roi.polygon)} 顶点)，请按 [w] 落盘写入文件"
                self.status_color = (0, 255, 0)
                print(f"[OK] 成功暂存 ROI: {new_roi.name} -> {new_roi.polygon}")
                self.current_pts.clear()

            # 重置当前正在画的点
            elif k == ord('r'):
                self.current_pts.clear()
                self.status_msg = "已清空当前正在绘制的点"
                self.status_color = (200, 200, 200)

            # 删除当前选中部件的标定
            elif k == ord('d'):
                cid = self.current_template["roi_id"]
                before_len = len(self.all_rois)
                self.all_rois = [r for r in self.all_rois if not (r.roi_id == cid and r.camera_id == self.camera_id)]
                if len(self.all_rois) < before_len:
                    self.status_msg = f"已删除部件: {cid}"
                    self.status_color = (0, 165, 255)
                    print(f"[*] 已删除 ROI: {cid}")
                else:
                    self.status_msg = f"未找到该部件的已有标定: {cid}"
                    self.status_color = (150, 150, 150)

            # 一键落盘写入文件
            elif k == ord('w'):
                ok = self.calibrator.save_rois(self.all_rois)
                if ok:
                    sync_info = ""
                    if self.sync_vehicle and os.path.exists(self.vehicle_calib_path):
                        ok_sync = self.calibrator.sync_to_vehicle_calib(self.all_rois, self.vehicle_calib_path)
                        sync_info = f" 并同步至 {self.vehicle_calib_path}" if ok_sync else " (vehicle_calib 同步失败)"
                    self.status_msg = f"【已落盘】成功写入 {self.config_path}{sync_info}"
                    self.status_color = (0, 255, 0)
                    print(f"[OK] 标定配置已成功落盘写入: {Path(self.config_path).resolve()}{sync_info}")
                else:
                    self.status_msg = "保存失败，请检查文件写入权限"
                    self.status_color = (0, 0, 255)

        cv2.destroyAllWindows()
        if self.cap:
            self.cap.release()

        # 退出前自动询问或保存
        if self.all_rois:
            self.calibrator.save_rois(self.all_rois)
            if self.sync_vehicle and os.path.exists(self.vehicle_calib_path):
                self.calibrator.sync_to_vehicle_calib(self.all_rois, self.vehicle_calib_path)
            print(f"[OK] 退出时已自动保存所有 ROI 标定数据 (共 {len(self.all_rois)} 项)。")


def parse_camera_id(val: str) -> int:
    """支持 0,1,2,3 数字或 front/left/right/rear 中英文别名"""
    val_str = str(val).strip().lower()
    mapping = {
        "0": 0, "front": 0, "f": 0, "前": 0, "车头": 0,
        "1": 1, "left": 1, "l": 1, "左": 1, "左侧": 1, "左视": 1,
        "2": 2, "right": 2, "r": 2, "右": 2, "右侧": 2, "右视": 2,
        "3": 3, "rear": 3, "back": 3, "b": 3, "后": 3, "车尾": 3, "后视": 3
    }
    if val_str in mapping:
        return mapping[val_str]
    try:
        cid = int(val_str)
        if cid in [0, 1, 2, 3]:
            return cid
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(f"无法识别相机通道: '{val}' (支持 0/1/2/3 或 front/left/right/rear)")


def main():
    parser = argparse.ArgumentParser(description="智能汽车露营守卫 - 车身与敏感部件固定 ROI 可视化标定工具")
    parser.add_argument("--image", type=str, default=None, help="静态鱼眼图像路径 (如 sample_fisheye.png)")
    parser.add_argument("--video", type=str, default=None, help="鱼眼视频路径 (如 sample_fisheye_left.mp4)")
    parser.add_argument("--camera-id", type=parse_camera_id, default=1,
                        help="挂载相机编号: 0前(front), 1左(left), 2右(right), 3后(rear) (默认 1 左侧相机)")
    parser.add_argument("--config", type=str, default="roi_config.json", help="ROI 标定保存路径 (默认: roi_config.json)")
    parser.add_argument("--vehicle-calib", type=str, default="vehicle_calib.json",
                        help="整车统一标定文件路径 (默认: vehicle_calib.json)")
    parser.add_argument("--no-sync", action="store_true", help="不自动同步到 vehicle_calib.json")

    args = parser.parse_args()

    # 1. 寻找合适的输入源
    image_path = args.image
    video_path = args.video
    frame = None
    cap = None

    if image_path and os.path.exists(image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[!] 无法读取图片: {image_path}")
            return
        print(f"[*] 成功载入标定图片: {image_path} ({frame.shape[1]}x{frame.shape[0]})")
    elif video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[!] 无法打开视频文件: {video_path}")
            return
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"[!] 无法从视频中读取首帧: {video_path}")
            return
        print(f"[*] 成功打开标定视频: {video_path} (按空格键可暂停选择画面)")
    else:
        # 尝试自动查找当前目录下可用的测试样本
        for candidate in ["sample_fisheye.png", "sample_fisheye.jpg", "sample_fisheye_left.mp4"]:
            if os.path.exists(candidate):
                if candidate.endswith(".mp4"):
                    cap = cv2.VideoCapture(candidate)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"[*] 未指定输入源，已自动选用预置视频样本: {candidate}")
                        break
                else:
                    frame = cv2.imread(candidate)
                    if frame is not None:
                        print(f"[*] 未指定输入源，已自动选用预置图片样本: {candidate}")
                        break

    if frame is None:
        print("[!] 错误: 未找到可用的标定底图或视频！")
        print("    请指定 --image <path.jpg> 或 --video <path.mp4>")
        return

    app = InteractiveCalibrateApp(
        base_frame=frame,
        camera_id=args.camera_id,
        config_path=args.config,
        vehicle_calib_path=args.vehicle_calib,
        sync_vehicle=not args.no_sync,
        cap=cap
    )
    app.run()


if __name__ == "__main__":
    main()
