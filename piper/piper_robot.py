#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from typing import List, Literal, Sequence

import numpy as np
from piper_sdk import C_PiperInterface_V2

# 对外统一的单位选项
PoseUnit = Literal["mmdeg", "mrad"]  # 末端位姿：毫米/角度 或 米/弧度
JointUnit = Literal["deg", "rad"]    # 关节角度：角度或弧度


class PiperRobot:
    """
    基于 quick_start 示例封装的易用机械臂控制类。
    - 位姿默认用毫米 + 角度（与 quick_start 保持一致）
    - 关节默认用弧度
    - 所有指令都会自动完成单位换算
    """

    def __init__(self, can_port: str = "can0", default_speed: int = 20):
        self.can_port = can_port
        self.default_speed = int(default_speed)
        self.piper = C_PiperInterface_V2(self.can_port)
        self._connected = False
        self._enabled = False

    # 基础生命周期 ---------------------------------------------------------
    def connect(self) -> None:
        if not self._connected:
            self.piper.ConnectPort()
            self._connected = True

    def enable(self, timeout_sec: float = 5.0) -> None:
        self.connect()
        start = time.time()
        while not self.piper.EnablePiper():
            if time.time() - start > timeout_sec:
                raise RuntimeError("Enable 机械臂超时")
            time.sleep(0.01)
        self._enabled = True

    def disable(self) -> None:
        if self._enabled:
            self.piper.DisablePiper()
            self._enabled = False

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.disable()

    # 内部工具 -------------------------------------------------------------
    def _ensure_enabled(self) -> None:
        if not self._enabled:
            raise RuntimeError("请先调用 enable() 完成使能")

    def _clamp_speed(self, speed: int | None) -> int:
        val = self.default_speed if speed is None else int(speed)
        return max(1, min(3000, val))

    def _pose_to_sdk(self, pose: Sequence[float], unit: PoseUnit) -> Sequence[int]:
        if len(pose) != 6:
            raise ValueError("pose 长度必须为 6，对应 [x, y, z, rx, ry, rz]")
        x, y, z, rx, ry, rz = pose
        if unit == "mmdeg":
            return (
                int(round(x * 1000)),   # mm -> μm
                int(round(y * 1000)),
                int(round(z * 1000)),
                int(round(rx * 1000)),  # deg -> mdeg
                int(round(ry * 1000)),
                int(round(rz * 1000)),
            )
        factor_pos = 1_000_000  # m -> μm
        factor_rot = 180.0 / np.pi * 1000  # rad -> mdeg
        return (
            int(round(x * factor_pos)),
            int(round(y * factor_pos)),
            int(round(z * factor_pos)),
            int(round(rx * factor_rot)),
            int(round(ry * factor_rot)),
            int(round(rz * factor_rot)),
        )

    def _joints_to_sdk(self, joints: Sequence[float], unit: JointUnit) -> Sequence[int]:
        if len(joints) != 6:
            raise ValueError("joints 长度必须为 6")
        if unit == "deg":
            factor = 1000.0  # deg -> mdeg
        else:
            factor = 180.0 / np.pi * 1000  # rad -> mdeg
        return [int(round(j * factor)) for j in joints]

    # 控制指令 -------------------------------------------------------------
    def movp(
        self,
        pose: Sequence[float],
        speed: int | None = None,
        unit: PoseUnit = "mmdeg",
    ) -> None:
        """
        末端笛卡尔控制（MOVE P）
        pose: [x, y, z, rx, ry, rz]
        unit:
          - "mmdeg": x/y/z 单位毫米, rx/ry/rz 单位度（与 quick_start 一致）
          - "mrad" : x/y/z 单位米,    rx/ry/rz 单位弧度
        """
        self._ensure_enabled()
        sdk_pose = self._pose_to_sdk(pose, unit)
        spd = self._clamp_speed(speed)
        self.piper.MotionCtrl_2(0x01, 0x00, spd, 0x00)
        self.piper.EndPoseCtrl(*sdk_pose)

    def movj(
        self,
        joints: Sequence[float],
        speed: int | None = None,
        unit: JointUnit = "rad",
        mit_mode: bool = False
    ) -> None:
        """
        关节空间控制（MOVE J）
        joints: 六个关节角
        unit: "rad"（默认）或 "deg"
        """
        self._ensure_enabled()
        sdk_joints = self._joints_to_sdk(joints, unit)
        spd = self._clamp_speed(speed)
        is_mit_mode = 0xAD if mit_mode else 0x00
        self.piper.ModeCtrl(0x01, 0x01, spd,0x00)
        self.piper.JointCtrl(*sdk_joints)

    def move_gripper(self, width_m: float, speed: int = 1000) -> None:
        """
        控制夹爪张开距离（毫米）。
        """
        self._ensure_enabled()
        target = max(0, width_m)
        gripper_pos = int(round(target * 1_000))  # m -> μm
        self.piper.GripperCtrl(gripper_pos, speed, 0x01, 0)

    def get_gripper(self):
        return self.piper.GetArmGripperMsgs()

    # 状态反馈 -------------------------------------------------------------
    def get_end_pose(self, unit: PoseUnit = "mmdeg") -> List[float]:
        msg = self.piper.GetArmEndPoseMsgs()
        if unit == "mmdeg":
            return [
                msg.end_pose.X_axis / 1000.0,
                msg.end_pose.Y_axis / 1000.0,
                msg.end_pose.Z_axis / 1000.0,
                msg.end_pose.RX_axis / 1000.0,
                msg.end_pose.RY_axis / 1000.0,
                msg.end_pose.RZ_axis / 1000.0,
            ]
        return [
            msg.end_pose.X_axis / 1_000_000.0,
            msg.end_pose.Y_axis / 1_000_000.0,
            msg.end_pose.Z_axis / 1_000_000.0,
            np.deg2rad(msg.end_pose.RX_axis / 1000.0),
            np.deg2rad(msg.end_pose.RY_axis / 1000.0),
            np.deg2rad(msg.end_pose.RZ_axis / 1000.0),
        ]

    def get_joint_angles(self, unit: JointUnit = "rad") -> List[float]:
        joint_msg = self.piper.GetArmJointMsgs()
        raw = [
            joint_msg.joint_state.joint_1,
            joint_msg.joint_state.joint_2,
            joint_msg.joint_state.joint_3,
            joint_msg.joint_state.joint_4,
            joint_msg.joint_state.joint_5,
            joint_msg.joint_state.joint_6,
        ]
        if unit == "deg":
            return [v / 1000.0 for v in raw]
        deg2rad = np.pi / 180.0 / 1000.0
        return [v * deg2rad for v in raw]

    # 便捷操作 -------------------------------------------------------------
    def go_zero(self, speed: int | None = None) -> None:
        """将六轴回零位。
        TCP_pose
        X: 200575, Y: 0, Z: 225906
        ArmMsgFeedBackEndPose:
            X_axis : 56127
            Y_axis : 0
            Z_axis : 213266
            RX_axis : 0
            RY_axis : 84999
            RZ_axis : 0
        """
        self.movj([0.0] * 6, speed=speed, unit="rad")

    def shutdown(self) -> None:
        """失能机械臂，适用于程序结束时的清理。"""
        self.disable()