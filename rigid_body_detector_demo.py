"""
智能汽车露营守卫系统 - 警戒线与微动抑制实时演示 Demo
功能特性：
  1. 画面划分清晰的警戒区（黄色/青色警戒线与半透明遮罩）
  2. 警戒线外的所有微动（晃动、走动、背景干扰）被【完全抑制】，不报警
  3. 警戒区内的微动被实时捕获，并结合刚体特征进行研判，触发红色声光高亮
  4. 支持按 '1' / '2' 键切换不同的警戒防区形态（地面透视梯形 / 水平分割线）
  5. 按 'r' 键重置背景，按 'q' 或 ESC 键退出
"""

import cv2
import numpy as np
import time

def open_camera():
    """打开本机摄像头"""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("【错误】未检测到可用摄像头或摄像头被占用！")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def get_zone_polygon(zone_mode=1, width=640, height=480):
    """生成警戒区多边形"""
    if zone_mode == 1:
        # 模式 1: 模拟车载视角的地面透视梯形防区 (前宽后窄，贴合地面透视)
        pts = np.array([
            [int(width * 0.22), int(height * 0.42)],  # 远端左
            [int(width * 0.78), int(height * 0.42)],  # 远端右
            [int(width * 0.94), int(height * 0.96)],  # 近端右
            [int(width * 0.06), int(height * 0.96)]   # 近端左
        ], dtype=np.int32)
    else:
        # 模式 2: 水平下半区警戒线 (画面下半部 50% 区域为警戒区)
        pts = np.array([
            [0, int(height * 0.50)],
            [width, int(height * 0.50)],
            [width, height],
            [0, height]
        ], dtype=np.int32)
    return pts

def main():
    cap = open_camera()
    if cap is None:
        return

    print("=" * 65)
    print("【露营守卫】警戒线与区外微动抑制演示已启动！")
    print("演示逻辑：")
    print("  - 警戒线外移动（画面上半部/远端）：将被【完全抑制】，不触发警报！")
    print("  - 警戒区内移动（发光边框内）：实时触发红框报警与微动/刚体研判！")
    print("操作按键：")
    print("  - 按 '1' 键：切换为【地面透视梯形警戒区】(默认)")
    print("  - 按 '2' 键：切换为【水平分割警戒区】")
    print("  - 按 'r' 键：重新校准背景模型")
    print("  - 按 'q' 或 ESC 键：退出程序")
    print("=" * 65)

    # 背景建模器
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    zone_mode = 1
    zone_poly = get_zone_polygon(zone_mode, 640, 480)

    prev_gray = None
    fps_counter = 0
    fps_time = time.time()
    current_fps = 0.0

    # 暖机前 15 帧收敛背景
    print("正在初始化环境背景，请保持镜头前 1 秒相对静止...")
    for _ in range(15):
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bg_subtractor.apply(gray)
            prev_gray = gray

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 统计 FPS
        fps_counter += 1
        now = time.time()
        if now - fps_time >= 1.0:
            current_fps = fps_counter / (now - fps_time)
            fps_counter = 0
            fps_time = now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 1. 背景差分提取全图移动前景
        fg_mask = bg_subtractor.apply(blurred)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, morph_kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_kernel)

        # 2. 检测前景连通域
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        in_zone_rigid = []       # 警戒区内：确认为刚体
        in_zone_non_rigid = []   # 警戒区内：非刚体微动 (如衣物摆动、手势晃动)
        suppressed_outside = []  # 警戒区外：微动已被完全抑制

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 800 or area > 180000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            # 核心接触点判定：取目标接地点 (底边中点)，同时结合中心点
            foot_pt = (int(x + w / 2), int(y + h))
            center_pt = (int(x + w / 2), int(y + h / 2))

            # 关键判定：判断目标接触点或中心点是否落在警戒区多边形内部
            is_inside = (cv2.pointPolygonTest(zone_poly, foot_pt, False) >= 0) or \
                        (cv2.pointPolygonTest(zone_poly, center_pt, False) >= 0)

            # ================= 核心逻辑：区外微动严格抑制 =================
            if not is_inside:
                # 警戒线外部的移动实体：记录但【完全抑制报警】
                suppressed_outside.append((x, y, w, h, foot_pt))
                continue

            # ================= 警戒区内部：启动微动与刚体研判 =================
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0.0

            # 提取 ROI 内部特征角点做运动学刚体校验
            roi_prev = prev_gray[y:y+h, x:x+w]
            pts_prev = cv2.goodFeaturesToTrack(roi_prev, maxCorners=30, qualityLevel=0.03, minDistance=6)

            is_rigid = False
            inlier_ratio = 0.0
            displacement = 0.0
            tracking_pts = None

            if pts_prev is not None and len(pts_prev) >= 6:
                pts_prev_full = pts_prev + np.array([x, y], dtype=np.float32)
                pts_curr_full, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, pts_prev_full, None,
                    winSize=(15, 15), maxLevel=2
                )

                if status is not None:
                    valid_prev = pts_prev_full[status.flatten() == 1].reshape(-1, 2)
                    valid_curr = pts_curr_full[status.flatten() == 1].reshape(-1, 2)

                    if len(valid_prev) >= 5:
                        M, inliers = cv2.estimateAffinePartial2D(
                            valid_prev, valid_curr,
                            method=cv2.RANSAC,
                            ransacReprojThreshold=2.5
                        )
                        if inliers is not None and len(inliers) > 0:
                            inliers = inliers.flatten()
                            inlier_ratio = float(np.sum(inliers)) / len(inliers)
                            displacement = float(np.linalg.norm(valid_curr.mean(axis=0) - valid_prev.mean(axis=0)))
                            tracking_pts = (valid_prev, valid_curr, inliers)

                            if solidity >= 0.55 and inlier_ratio >= 0.65 and displacement >= 1.2:
                                is_rigid = True

            if is_rigid:
                in_zone_rigid.append({
                    "box": (x, y, w, h),
                    "foot": foot_pt,
                    "solidity": solidity,
                    "inlier": inlier_ratio,
                    "disp": displacement,
                    "pts": tracking_pts
                })
            else:
                in_zone_non_rigid.append((x, y, w, h, foot_pt, solidity))

        prev_gray = gray.copy()

        # ========================= 界面可视化渲染 =========================
        # 1. 警戒区底色与边界绘制
        overlay = frame.copy()
        has_in_zone_motion = (len(in_zone_rigid) > 0 or len(in_zone_non_rigid) > 0)

        # 若警戒区内有微动：警戒区高亮偏红报警；若无微动：警戒区呈现安全青绿色
        zone_color = (0, 0, 255) if has_in_zone_motion else (0, 255, 200)
        border_color = (0, 0, 255) if has_in_zone_motion else (0, 220, 255)
        border_thickness = 3 if has_in_zone_motion else 2

        # 绘制半透明警戒区遮罩
        cv2.fillPoly(overlay, [zone_poly], zone_color)
        alpha = 0.20 if not has_in_zone_motion else 0.30
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # 绘制警戒线实线与发光顶点
        cv2.polylines(frame, [zone_poly], isClosed=True, color=border_color, thickness=border_thickness)
        for pt in zone_poly:
            cv2.circle(frame, tuple(pt), 4, border_color, -1)

        # 标注警戒线名称
        label_pos = (zone_poly[0][0] + 10, zone_poly[0][1] - 8)
        cv2.putText(frame, "=== WARNING ZONE BOUNDARY ===", label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, border_color, 2)

        # 2. 绘制【警戒线外被抑制的微动】（暗淡灰色虚线框，提示已抑制）
        for (sx, sy, sw, sh, s_foot) in suppressed_outside:
            cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (100, 100, 100), 1)
            cv2.putText(frame, "[OUTSIDE: SUPPRESSED]", (sx, sy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1)
            cv2.circle(frame, s_foot, 3, (120, 120, 120), -1)

        # 3. 绘制【警戒区内检测到的非刚体微动】（黄色警告框）
        for (nx, ny, nw, nh, n_foot, n_sol) in in_zone_non_rigid:
            cv2.rectangle(frame, (nx, ny), (nx + nw, ny + nh), (0, 215, 255), 2)
            cv2.putText(frame, f"IN-ZONE: Motion (Solid:{n_sol:.2f})", (nx, ny - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 215, 255), 2)
            cv2.circle(frame, n_foot, 4, (0, 215, 255), -1)

        # 4. 绘制【警戒区内检测到的刚体入侵】（红色高危框 + 运动轨迹点）
        for item in in_zone_rigid:
            bx, by, bw, bh = item["box"]
            foot = item["foot"]
            inl = item["inlier"]
            disp = item["disp"]
            v_prev, v_curr, inliers = item["pts"]

            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 3)
            # 绘制接地点
            cv2.circle(frame, foot, 5, (0, 0, 255), -1)

            # 绘制刚体特征轨迹
            if inliers is not None:
                for i, is_inlier in enumerate(inliers):
                    p1 = (int(round(v_prev[i, 0])), int(round(v_prev[i, 1])))
                    p2 = (int(round(v_curr[i, 0])), int(round(v_curr[i, 1])))
                    if is_inlier == 1:
                        cv2.circle(frame, p2, 3, (0, 255, 0), -1)
                        cv2.line(frame, p1, p2, (0, 255, 0), 2)

            info_text = f"ALERT: IN-ZONE RIGID (Inlier:{inl*100:.0f}%, Mov:{disp:.1f}px)"
            cv2.putText(frame, info_text, (bx, by - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2)

        # 5. 顶部系统实时状态大横幅
        cv2.rectangle(frame, (0, 0), (640, 40), (15, 15, 15), -1)
        if len(in_zone_rigid) > 0:
            status_text = f"[!] CRITICAL ALARM: {len(in_zone_rigid)} RIGID INTRUDER IN ZONE!"
            banner_color = (0, 0, 255)
        elif len(in_zone_non_rigid) > 0:
            status_text = f"[?] WARNING: {len(in_zone_non_rigid)} UNKNOWN MOTION IN ZONE"
            banner_color = (0, 215, 255)
        else:
            status_text = "[OK] ZONE SAFE | Outside Motion Suppressed"
            banner_color = (0, 255, 0)

        cv2.putText(frame, status_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, banner_color, 2)
        cv2.putText(frame, f"FPS: {current_fps:.1f} | Key: '1'/'2' Zone, 'r' Reset", (315, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

        # 6. 右下角微型雷达状态指示
        cv2.rectangle(frame, (450, 405), (635, 475), (20, 20, 20), -1)
        cv2.rectangle(frame, (450, 405), (635, 475), (80, 80, 80), 1)
        cv2.putText(frame, f"In-Zone Alerts : {len(in_zone_rigid) + len(in_zone_non_rigid)}", (458, 428),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255) if has_in_zone_motion else (0, 255, 0), 1)
        cv2.putText(frame, f"Suppressed Out : {len(suppressed_outside)}", (458, 452),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        # 显示主画面
        cv2.imshow("Camping Guard - Warning Zone & Suppression Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)
            print(">> 背景模型已重新校准！")
        elif key == ord('1'):
            zone_mode = 1
            zone_poly = get_zone_polygon(zone_mode, 640, 480)
            print(">> 已切换为模式 1：【地面透视梯形警戒区】")
        elif key == ord('2'):
            zone_mode = 2
            zone_poly = get_zone_polygon(zone_mode, 640, 480)
            print(">> 已切换为模式 2：【水平分割警戒区】")

    cap.release()
    cv2.destroyAllWindows()
    print("演示已退出。")

if __name__ == "__main__":
    main()
