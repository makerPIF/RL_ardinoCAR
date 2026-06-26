#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
순환 주행 강화학습 시뮬레이터 (v3)
- 중앙 장애물 크기 조절 가능
- 도로 폭 조절 가능
- 한글 폰트 지원

호박공장메이커스페이스 - Student Rover Challenge Korea
"""

import sys
import os
import json
import math
import random
import traceback
import numpy as np
from collections import deque

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QGroupBox, QTextEdit, QProgressBar, QFileDialog, QMessageBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QComboBox, QSlider
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import (
    QPainter, QColor, QBrush, QPen, QFont, QLinearGradient,
    QFontDatabase
)

# Matplotlib 한글 폰트 설정
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 한글 폰트 설정 (Windows: Malgun Gothic, Mac: AppleGothic, Linux: NanumGothic)
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':  # Mac
    plt.rcParams['font.family'] = 'AppleGothic'
else:  # Linux
    plt.rcParams['font.family'] = 'NanumGothic'

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


# ============================================================
# 환경 클래스 - 순환 도로 환경
# ============================================================
class CircuitEnvironment:
    """순환 도로 환경"""
    
    SCALE = 5  # 1cm = 5픽셀
    
    def __init__(self, obstacle_size_cm=30, road_width_cm=40):
        # 장애물 크기와 도로폭 (cm)
        self.obstacle_size_cm = obstacle_size_cm
        self.road_width_cm = road_width_cm
        
        self.update_dimensions()
        self.reset()
    
    def update_dimensions(self):
        """환경 크기 계산"""
        # 전체 크기 = 장애물 + 양쪽 도로
        total_size_cm = self.obstacle_size_cm + (self.road_width_cm * 2)
        
        self.width = total_size_cm * self.SCALE
        self.height = total_size_cm * self.SCALE
        
        # 중앙 장애물 (픽셀)
        self.obstacle_size = self.obstacle_size_cm * self.SCALE
        self.obstacle_x = self.width // 2
        self.obstacle_y = self.height // 2
        
        # 도로 폭 (픽셀)
        self.road_width = self.road_width_cm * self.SCALE
        
        # 자동차 크기 (픽셀)
        self.car_length = 15 * self.SCALE
        self.car_width = 10 * self.SCALE
        
        # 센서 최대 범위 (픽셀)
        self.sensor_range = 50 * self.SCALE
        
        # 체크포인트 (순환 확인용)
        self.checkpoints = [
            (self.width // 2, self.road_width // 2),
            (self.width - self.road_width // 2, self.height // 2),
            (self.width // 2, self.height - self.road_width // 2),
            (self.road_width // 2, self.height // 2),
        ]
    
    def set_obstacle_size(self, size_cm):
        """장애물 크기 변경"""
        self.obstacle_size_cm = size_cm
        self.update_dimensions()
    
    def set_road_width(self, width_cm):
        """도로폭 변경"""
        self.road_width_cm = width_cm
        self.update_dimensions()
    
    def reset(self):
        """환경 초기화"""
        self.car_x = float(self.width // 2)
        self.car_y = float(self.height - self.road_width // 2)
        self.car_angle = -90.0
        
        self.steps = 0
        self.max_steps = 1000
        
        self.checkpoint_passed = [False, False, False, False]
        self.last_checkpoint = -1
        self.laps_completed = 0
        
        self.done = False
        self.collision = False
        
        return self.get_state()
    
    def get_sensor_readings(self):
        """3개의 초음파 센서 값 반환 (cm)"""
        readings = []
        sensor_angles = [-45, 0, 45]
        
        for angle_offset in sensor_angles:
            angle_rad = math.radians(self.car_angle + angle_offset)
            distance = self.sensor_range
            
            for d in range(5, self.sensor_range + 1, 5):
                test_x = self.car_x + d * math.cos(angle_rad)
                test_y = self.car_y + d * math.sin(angle_rad)
                
                # 외벽 충돌 체크
                if test_x < 0 or test_x > self.width or test_y < 0 or test_y > self.height:
                    distance = d
                    break
                
                # 중앙 장애물 충돌 체크
                half_obs = self.obstacle_size // 2
                if (self.obstacle_x - half_obs <= test_x <= self.obstacle_x + half_obs and
                    self.obstacle_y - half_obs <= test_y <= self.obstacle_y + half_obs):
                    distance = d
                    break
            
            readings.append(distance / self.SCALE)
        
        return readings
    
    def discretize_distance(self, distance_cm):
        """거리 이산화 (5단계)"""
        if distance_cm < 10:
            return 0
        elif distance_cm < 20:
            return 1
        elif distance_cm < 30:
            return 2
        elif distance_cm < 40:
            return 3
        else:
            return 4
    
    def get_state(self):
        """현재 상태 반환"""
        readings = self.get_sensor_readings()
        state = tuple(self.discretize_distance(r) for r in readings)
        return state
    
    def step(self, action):
        """행동 실행"""
        self.steps += 1
        old_x, old_y = self.car_x, self.car_y
        
        move_speed = 6
        turn_angle = 8
        sharp_turn = 15
        
        if action == 0:  # 전진
            pass
        elif action == 1:  # 좌회전
            self.car_angle -= turn_angle
        elif action == 2:  # 우회전
            self.car_angle += turn_angle
        elif action == 3:  # 급좌회전
            self.car_angle -= sharp_turn
            move_speed = 4
        elif action == 4:  # 급우회전
            self.car_angle += sharp_turn
            move_speed = 4
        
        self.car_angle = self.car_angle % 360
        
        angle_rad = math.radians(self.car_angle)
        self.car_x += move_speed * math.cos(angle_rad)
        self.car_y += move_speed * math.sin(angle_rad)
        
        distance_moved = math.sqrt((self.car_x - old_x)**2 + (self.car_y - old_y)**2)
        
        collision = self.check_collision()
        checkpoint_reward = self.check_checkpoints()
        reward = self.calculate_reward(collision, checkpoint_reward, distance_moved)
        
        if collision:
            self.done = True
            self.collision = True
        elif self.steps >= self.max_steps:
            self.done = True
        
        return self.get_state(), reward, self.done
    
    def check_collision(self):
        """충돌 체크"""
        car_radius = min(self.car_length, self.car_width) // 2
        
        if (self.car_x - car_radius < 0 or 
            self.car_x + car_radius > self.width or
            self.car_y - car_radius < 0 or 
            self.car_y + car_radius > self.height):
            return True
        
        half_obs = self.obstacle_size // 2
        closest_x = max(self.obstacle_x - half_obs, 
                       min(self.car_x, self.obstacle_x + half_obs))
        closest_y = max(self.obstacle_y - half_obs, 
                       min(self.car_y, self.obstacle_y + half_obs))
        
        dist_to_obstacle = math.sqrt((self.car_x - closest_x)**2 + 
                                     (self.car_y - closest_y)**2)
        
        if dist_to_obstacle < car_radius:
            return True
        
        return False
    
    def check_checkpoints(self):
        """체크포인트 통과 확인"""
        reward = 0
        checkpoint_radius = 60
        
        for i, (cx, cy) in enumerate(self.checkpoints):
            dist = math.sqrt((self.car_x - cx)**2 + (self.car_y - cy)**2)
            
            if dist < checkpoint_radius and not self.checkpoint_passed[i]:
                expected_next = (self.last_checkpoint + 1) % 4
                if i == expected_next:
                    self.checkpoint_passed[i] = True
                    self.last_checkpoint = i
                    reward = 20
                    
                    if all(self.checkpoint_passed):
                        self.laps_completed += 1
                        self.checkpoint_passed = [False, False, False, False]
                        reward = 100
        
        return reward
    
    def calculate_reward(self, collision, checkpoint_reward, distance_moved):
        """보상 계산"""
        if collision:
            return -100
        
        reward = checkpoint_reward
        reward += distance_moved * 0.1
        
        readings = self.get_sensor_readings()
        left_dist = readings[0]
        right_dist = readings[2]
        
        balance = abs(left_dist - right_dist)
        if balance < 5:
            reward += 1
        elif balance < 10:
            reward += 0.5
        
        min_dist = min(readings)
        if min_dist < 10:
            reward -= 5
        elif min_dist < 15:
            reward -= 2
        
        return reward


# ============================================================
# Q-Learning 에이전트
# ============================================================
class QLearningAgent:
    """Q-Learning 에이전트"""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.n_states = 125
        self.n_actions = 5
        
        self.q_table = np.zeros((self.n_states, self.n_actions))
        
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_laps = []
        self.epsilon_history = []
    
    def get_state_index(self, state):
        return state[0] * 25 + state[1] * 5 + state[2]
    
    def choose_action(self, state, training=True):
        state_idx = self.get_state_index(state)
        
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return int(np.argmax(self.q_table[state_idx]))
    
    def learn(self, state, action, reward, next_state, done):
        state_idx = self.get_state_index(state)
        next_state_idx = self.get_state_index(next_state)
        
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state_idx])
        
        self.q_table[state_idx, action] += self.learning_rate * (
            target - self.q_table[state_idx, action]
        )
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_policy_table(self):
        policy = {}
        for state_idx in range(self.n_states):
            left = state_idx // 25
            center = (state_idx % 25) // 5
            right = state_idx % 5
            action = int(np.argmax(self.q_table[state_idx]))
            q_value = float(np.max(self.q_table[state_idx]))
            policy[(left, center, right)] = (action, q_value)
        return policy
    
    def save_model(self, filename):
        data = {
            'q_table': self.q_table.tolist(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_laps': self.episode_laps,
            'parameters': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon_decay': self.epsilon_decay
            }
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_model(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.q_table = np.array(data['q_table'])
        self.epsilon = data.get('epsilon', 0.01)
        self.episode_rewards = data.get('episode_rewards', [])
        self.episode_laps = data.get('episode_laps', [])


# ============================================================
# 시뮬레이션 캔버스
# ============================================================
class SimulationCanvas(QWidget):
    """시뮬레이션 시각화 캔버스"""
    
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.update_size()
        self.setStyleSheet("background-color: #1a1a2e;")
    
    def update_size(self):
        """캔버스 크기 업데이트"""
        size = max(400, min(550, self.env.width))
        self.setMinimumSize(size, size)
        self.setMaximumSize(size, size)
        self.scale_factor = size / self.env.width if self.env.width > 0 else 1
    
    def paintEvent(self, event):
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            sf = self.scale_factor
            
            # 배경
            painter.fillRect(self.rect(), QColor(26, 26, 46))
            
            # 도로 (회색)
            road_color = QColor(80, 80, 100)
            painter.setBrush(QBrush(road_color))
            painter.setPen(QPen(road_color))
            painter.drawRect(0, 0, int(self.env.width * sf), int(self.env.height * sf))
            
            # 중앙 장애물
            half_obs = self.env.obstacle_size // 2
            ox = int((self.env.obstacle_x - half_obs) * sf)
            oy = int((self.env.obstacle_y - half_obs) * sf)
            ow = int(self.env.obstacle_size * sf)
            oh = int(self.env.obstacle_size * sf)
            
            gradient = QLinearGradient(ox, oy, ox + ow, oy + oh)
            gradient.setColorAt(0, QColor(231, 76, 60))
            gradient.setColorAt(1, QColor(192, 57, 43))
            painter.setBrush(QBrush(gradient))
            painter.setPen(QPen(QColor(150, 40, 30), 3))
            painter.drawRect(ox, oy, ow, oh)
            
            # 장애물 텍스트
            painter.setPen(QPen(Qt.white))
            painter.setFont(QFont('Malgun Gothic', 10, QFont.Bold))
            painter.drawText(QRectF(ox, oy + oh//2 - 15, ow, 20), Qt.AlignCenter, "장애물")
            painter.setFont(QFont('Malgun Gothic', 8))
            painter.drawText(QRectF(ox, oy + oh//2, ow, 20), Qt.AlignCenter, 
                           f"{self.env.obstacle_size_cm}x{self.env.obstacle_size_cm}cm")
            
            # 도로 중앙선
            pen = QPen(QColor(255, 255, 255, 100), 2, Qt.DashLine)
            painter.setPen(pen)
            road_center = int(self.env.road_width * sf // 2)
            w = int(self.env.width * sf)
            h = int(self.env.height * sf)
            obs_half = int(half_obs * sf)
            
            painter.drawLine(obs_half + 30, road_center, w - obs_half - 30, road_center)
            painter.drawLine(obs_half + 30, h - road_center, w - obs_half - 30, h - road_center)
            painter.drawLine(road_center, obs_half + 30, road_center, h - obs_half - 30)
            painter.drawLine(w - road_center, obs_half + 30, w - road_center, h - obs_half - 30)
            
            # 체크포인트
            for i, (cx, cy) in enumerate(self.env.checkpoints):
                cx_s = int(cx * sf)
                cy_s = int(cy * sf)
                
                if self.env.checkpoint_passed[i]:
                    color = QColor(46, 204, 113, 150)
                else:
                    color = QColor(241, 196, 15, 100)
                
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(color.darker(), 2))
                painter.drawEllipse(QPointF(cx_s, cy_s), 20, 20)
                
                painter.setPen(QPen(Qt.white))
                painter.setFont(QFont('Arial', 9, QFont.Bold))
                painter.drawText(QRectF(cx_s - 10, cy_s - 7, 20, 14), Qt.AlignCenter, str(i + 1))
            
            # 센서 레이
            readings = self.env.get_sensor_readings()
            sensor_angles = [-45, 0, 45]
            
            for i, (angle_offset, reading) in enumerate(zip(sensor_angles, readings)):
                angle_rad = math.radians(self.env.car_angle + angle_offset)
                distance_px = reading * self.env.SCALE
                
                end_x = self.env.car_x + distance_px * math.cos(angle_rad)
                end_y = self.env.car_y + distance_px * math.sin(angle_rad)
                
                if reading < 10:
                    color = QColor(255, 0, 0, 180)
                elif reading < 20:
                    color = QColor(255, 165, 0, 150)
                elif reading < 30:
                    color = QColor(255, 255, 0, 120)
                else:
                    color = QColor(0, 255, 0, 100)
                
                painter.setPen(QPen(color, 2, Qt.DashLine))
                painter.drawLine(int(self.env.car_x * sf), int(self.env.car_y * sf),
                               int(end_x * sf), int(end_y * sf))
                
                painter.setBrush(QBrush(color))
                painter.drawEllipse(QPointF(end_x * sf, end_y * sf), 4, 4)
            
            # 자동차
            painter.save()
            painter.translate(self.env.car_x * sf, self.env.car_y * sf)
            painter.rotate(self.env.car_angle + 90)
            
            half_length = int(self.env.car_length * sf // 2)
            half_width = int(self.env.car_width * sf // 2)
            car_w = int(self.env.car_width * sf)
            car_l = int(self.env.car_length * sf)
            
            gradient = QLinearGradient(-half_width, -half_length, half_width, half_length)
            gradient.setColorAt(0, QColor(52, 152, 219))
            gradient.setColorAt(1, QColor(41, 128, 185))
            painter.setBrush(QBrush(gradient))
            painter.setPen(QPen(QColor(30, 100, 150), 2))
            painter.drawRoundedRect(-half_width, -half_length, car_w, car_l, 6, 6)
            
            painter.setBrush(QBrush(QColor(241, 196, 15)))
            painter.setPen(Qt.NoPen)
            painter.drawRect(-half_width + 4, -half_length, car_w - 8, 10)
            
            wheel_color = QColor(44, 62, 80)
            painter.setBrush(QBrush(wheel_color))
            painter.drawRect(-half_width - 4, -half_length + 8, 6, 15)
            painter.drawRect(half_width - 2, -half_length + 8, 6, 15)
            painter.drawRect(-half_width - 4, half_length - 23, 6, 15)
            painter.drawRect(half_width - 2, half_length - 23, 6, 15)
            
            painter.restore()
            
            # 정보 표시
            painter.setPen(QPen(Qt.white))
            painter.setFont(QFont('Malgun Gothic', 9))
            painter.drawText(10, 18, f"센서 (좌/중/우): {readings[0]:.0f} / {readings[1]:.0f} / {readings[2]:.0f} cm")
            
            state = self.env.get_state()
            painter.drawText(10, 36, f"상태: ({state[0]}, {state[1]}, {state[2]})")
            painter.drawText(10, 54, f"스텝: {self.env.steps} | 완주: {self.env.laps_completed}바퀴")
            
            total_cm = self.env.obstacle_size_cm + self.env.road_width_cm * 2
            painter.drawText(10, int(self.env.height * sf) - 8, 
                           f"환경: {total_cm}x{total_cm}cm | 도로폭: {self.env.road_width_cm}cm")
            
        except Exception as e:
            print(f"Paint error: {e}")


# ============================================================
# 학습 스레드
# ============================================================
class TrainingThread(QThread):
    """백그라운드 학습 스레드"""
    
    progress_signal = pyqtSignal(int, float, float, int, float)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    update_env_signal = pyqtSignal(float, float, float, int, int, list)
    
    def __init__(self, agent, episodes, obstacle_size_cm, road_width_cm, visualize=False):
        super().__init__()
        self.env = CircuitEnvironment(obstacle_size_cm, road_width_cm)
        self.agent = agent
        self.episodes = episodes
        self.visualize = visualize
        self.running = True
        self.paused = False
    
    def run(self):
        try:
            recent_rewards = deque(maxlen=50)
            
            for episode in range(self.episodes):
                if not self.running:
                    break
                
                while self.paused and self.running:
                    self.msleep(100)
                
                if not self.running:
                    break
                
                state = self.env.reset()
                total_reward = 0
                done = False
                
                while not done and self.running:
                    while self.paused and self.running:
                        self.msleep(100)
                    
                    if not self.running:
                        break
                    
                    action = self.agent.choose_action(state, training=True)
                    next_state, reward, done = self.env.step(action)
                    self.agent.learn(state, action, reward, next_state, done)
                    
                    state = next_state
                    total_reward += reward
                    
                    if self.visualize:
                        self.update_env_signal.emit(
                            self.env.car_x, self.env.car_y, self.env.car_angle,
                            self.env.steps, self.env.laps_completed,
                            list(self.env.checkpoint_passed)
                        )
                        self.msleep(20)
                
                if not self.running:
                    break
                
                self.agent.decay_epsilon()
                self.agent.episode_rewards.append(total_reward)
                self.agent.episode_laps.append(self.env.laps_completed)
                self.agent.epsilon_history.append(self.agent.epsilon)
                
                recent_rewards.append(total_reward)
                avg_reward = sum(recent_rewards) / len(recent_rewards)
                
                self.progress_signal.emit(
                    episode + 1, total_reward, self.agent.epsilon,
                    self.env.laps_completed, avg_reward
                )
            
            self.finished_signal.emit()
            
        except Exception as e:
            error_msg = f"학습 오류: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.error_signal.emit(error_msg)
            self.finished_signal.emit()
    
    def stop(self):
        self.running = False
    
    def toggle_pause(self):
        self.paused = not self.paused


# ============================================================
# 그래프 위젯 (한글 지원)
# ============================================================
class GraphWidget(FigureCanvas):
    """학습 그래프 (한글 폰트 지원)"""
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 6), facecolor='#2D2D2D')
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.axes = []
        for i in range(4):
            ax = self.fig.add_subplot(2, 2, i + 1)
            ax.set_facecolor('#2D2D2D')
            ax.tick_params(colors='white', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('white')
            self.axes.append(ax)
        
        self.fig.tight_layout(pad=2.0)
    
    def update_graphs(self, rewards, laps, epsilon_history):
        """그래프 업데이트"""
        try:
            for ax in self.axes:
                ax.clear()
                ax.set_facecolor('#2D2D2D')
                ax.tick_params(colors='white', labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color('white')
            
            if not rewards:
                self.draw()
                return
            
            episodes = list(range(1, len(rewards) + 1))
            
            # 1. 보상 그래프
            if len(rewards) >= 20:
                window = min(50, len(rewards))
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                self.axes[0].plot(range(window, len(rewards) + 1), moving_avg,
                                 color='#3498DB', linewidth=2, label='Moving Avg')
            self.axes[0].plot(episodes, rewards, color='#3498DB', alpha=0.3, linewidth=0.5)
            self.axes[0].set_title('Episode Reward', color='white', fontsize=10)
            self.axes[0].set_xlabel('Episode', color='white', fontsize=8)
            self.axes[0].legend(loc='lower right', fontsize=7, facecolor='#2D2D2D', 
                               labelcolor='white')
            
            # 2. 완주 횟수
            if laps:
                self.axes[1].bar(episodes, laps, color='#2ECC71', alpha=0.7)
                if len(laps) >= 20:
                    window = min(20, len(laps))
                    moving_avg = np.convolve(laps, np.ones(window)/window, mode='valid')
                    self.axes[1].plot(range(window, len(laps) + 1), moving_avg,
                                     color='#E74C3C', linewidth=2)
            self.axes[1].set_title('Laps Completed', color='white', fontsize=10)
            self.axes[1].set_xlabel('Episode', color='white', fontsize=8)
            self.axes[1].set_ylabel('Laps', color='white', fontsize=8)
            
            # 3. 탐험률
            if epsilon_history:
                self.axes[2].plot(episodes, epsilon_history, color='#F39C12', linewidth=2)
            self.axes[2].set_title('Epsilon (Exploration)', color='white', fontsize=10)
            self.axes[2].set_xlabel('Episode', color='white', fontsize=8)
            self.axes[2].set_ylim(0, 1)
            
            # 4. 누적 보상
            if rewards:
                cumsum = np.cumsum(rewards)
                self.axes[3].fill_between(episodes, cumsum, alpha=0.3, color='#9B59B6')
                self.axes[3].plot(episodes, cumsum, color='#9B59B6', linewidth=2)
            self.axes[3].set_title('Cumulative Reward', color='white', fontsize=10)
            self.axes[3].set_xlabel('Episode', color='white', fontsize=8)
            
            self.fig.tight_layout(pad=2.0)
            self.draw()
            
        except Exception as e:
            print(f"Graph update error: {e}")


# ============================================================
# 메인 윈도우
# ============================================================
class MainWindow(QMainWindow):
    """메인 애플리케이션"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("순환 주행 AI - 강화학습 시뮬레이터")
        self.setMinimumSize(1350, 800)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #1E1E1E; }
            QWidget { color: white; }
            QLabel { font-size: 11px; font-family: 'Malgun Gothic'; }
            QGroupBox {
                color: white;
                border: 2px solid #3498DB;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 10px;
                font-weight: bold;
                font-family: 'Malgun Gothic';
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 11px;
                font-family: 'Malgun Gothic';
            }
            QPushButton:hover { background-color: #2980B9; }
            QPushButton:pressed { background-color: #1F618D; }
            QPushButton:disabled { background-color: #555; color: #888; }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #2D2D2D;
                color: white;
                border: 1px solid #3498DB;
                border-radius: 4px;
                padding: 5px;
                min-width: 70px;
                font-family: 'Malgun Gothic';
            }
            QTextEdit {
                background-color: #2D2D2D;
                color: #2ECC71;
                border: 1px solid #444;
                border-radius: 4px;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 10px;
            }
            QProgressBar {
                border: 1px solid #3498DB;
                border-radius: 5px;
                text-align: center;
                color: white;
                background-color: #2D2D2D;
                font-family: 'Malgun Gothic';
            }
            QProgressBar::chunk {
                background-color: #2ECC71;
                border-radius: 4px;
            }
            QTabWidget::pane {
                border: 1px solid #3498DB;
                border-radius: 4px;
                background-color: #252525;
            }
            QTabBar::tab {
                background-color: #2D2D2D;
                color: white;
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
                font-family: 'Malgun Gothic';
            }
            QTabBar::tab:selected {
                background-color: #3498DB;
            }
            QTableWidget {
                background-color: #2D2D2D;
                color: white;
                gridline-color: #444;
                border: none;
                font-family: 'Malgun Gothic';
            }
            QTableWidget::item { padding: 4px; }
            QHeaderView::section {
                background-color: #3498DB;
                color: white;
                padding: 6px;
                border: none;
                font-weight: bold;
                font-family: 'Malgun Gothic';
            }
            QSlider::groove:horizontal {
                border: 1px solid #3498DB;
                height: 8px;
                background: #2D2D2D;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498DB;
                border: 1px solid #2980B9;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #2980B9;
            }
        """)
        
        # 환경 설정 기본값
        self.obstacle_size_cm = 30
        self.road_width_cm = 40
        
        self.env = CircuitEnvironment(self.obstacle_size_cm, self.road_width_cm)
        self.agent = QLearningAgent()
        self.training_thread = None
        
        self.init_ui()
        
        self.test_timer = QTimer()
        self.test_timer.timeout.connect(self.test_step)
        self.testing = False
        
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.canvas.update)
        self.update_timer.start(50)
    
    def init_ui(self):
        """UI 초기화"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(10)
        
        # === 왼쪽 패널 ===
        left_panel = QVBoxLayout()
        left_panel.setSpacing(8)
        
        # 시뮬레이션 캔버스
        canvas_group = QGroupBox("시뮬레이션")
        canvas_layout = QVBoxLayout(canvas_group)
        self.canvas = SimulationCanvas(self.env)
        canvas_layout.addWidget(self.canvas)
        left_panel.addWidget(canvas_group)
        
        # ========== 환경 설정 그룹 ==========
        env_group = QGroupBox("환경 설정")
        env_layout = QGridLayout(env_group)
        env_layout.setSpacing(8)
        
        # 장애물 크기
        env_layout.addWidget(QLabel("장애물 크기:"), 0, 0)
        self.obstacle_spin = QSpinBox()
        self.obstacle_spin.setRange(10, 60)
        self.obstacle_spin.setValue(self.obstacle_size_cm)
        self.obstacle_spin.setSuffix(" cm")
        self.obstacle_spin.valueChanged.connect(self.on_obstacle_changed)
        env_layout.addWidget(self.obstacle_spin, 0, 1)
        
        self.obstacle_slider = QSlider(Qt.Horizontal)
        self.obstacle_slider.setRange(10, 60)
        self.obstacle_slider.setValue(self.obstacle_size_cm)
        self.obstacle_slider.valueChanged.connect(self.obstacle_spin.setValue)
        self.obstacle_spin.valueChanged.connect(self.obstacle_slider.setValue)
        env_layout.addWidget(self.obstacle_slider, 0, 2)
        
        # 도로 폭
        env_layout.addWidget(QLabel("도로 폭:"), 1, 0)
        self.road_spin = QSpinBox()
        self.road_spin.setRange(20, 60)
        self.road_spin.setValue(self.road_width_cm)
        self.road_spin.setSuffix(" cm")
        self.road_spin.valueChanged.connect(self.on_road_changed)
        env_layout.addWidget(self.road_spin, 1, 1)
        
        self.road_slider = QSlider(Qt.Horizontal)
        self.road_slider.setRange(20, 60)
        self.road_slider.setValue(self.road_width_cm)
        self.road_slider.valueChanged.connect(self.road_spin.setValue)
        self.road_spin.valueChanged.connect(self.road_slider.setValue)
        env_layout.addWidget(self.road_slider, 1, 2)
        
        # 환경 크기 표시
        self.env_info_label = QLabel()
        self.update_env_info_label()
        self.env_info_label.setStyleSheet("color: #888; font-size: 10px;")
        env_layout.addWidget(self.env_info_label, 2, 0, 1, 3)
        
        left_panel.addWidget(env_group)
        
        # 학습 파라미터 그룹
        param_group = QGroupBox("학습 파라미터")
        param_layout = QGridLayout(param_group)
        param_layout.setSpacing(8)
        
        param_layout.addWidget(QLabel("학습 횟수:"), 0, 0)
        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(10, 10000)
        self.episodes_spin.setValue(500)
        self.episodes_spin.setSingleStep(100)
        param_layout.addWidget(self.episodes_spin, 0, 1)
        
        param_layout.addWidget(QLabel("학습률 (α):"), 0, 2)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.01, 1.0)
        self.lr_spin.setValue(0.15)
        self.lr_spin.setSingleStep(0.05)
        self.lr_spin.setDecimals(2)
        param_layout.addWidget(self.lr_spin, 0, 3)
        
        param_layout.addWidget(QLabel("할인율 (γ):"), 1, 0)
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.5, 0.99)
        self.gamma_spin.setValue(0.95)
        self.gamma_spin.setSingleStep(0.05)
        self.gamma_spin.setDecimals(2)
        param_layout.addWidget(self.gamma_spin, 1, 1)
        
        param_layout.addWidget(QLabel("ε 감소율:"), 1, 2)
        self.decay_spin = QDoubleSpinBox()
        self.decay_spin.setRange(0.9, 0.999)
        self.decay_spin.setValue(0.995)
        self.decay_spin.setSingleStep(0.002)
        self.decay_spin.setDecimals(3)
        param_layout.addWidget(self.decay_spin, 1, 3)
        
        param_layout.addWidget(QLabel("시각화:"), 2, 0)
        self.visualize_combo = QComboBox()
        self.visualize_combo.addItems(["끔 (빠른 학습)", "켬 (시각화)"])
        param_layout.addWidget(self.visualize_combo, 2, 1)
        
        left_panel.addWidget(param_group)
        
        # 버튼들
        btn_layout1 = QHBoxLayout()
        self.start_btn = QPushButton("▶ 학습 시작")
        self.start_btn.clicked.connect(self.start_training)
        btn_layout1.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("⏸ 일시정지")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)
        btn_layout1.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("⏹ 중지")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        btn_layout1.addWidget(self.stop_btn)
        left_panel.addLayout(btn_layout1)
        
        btn_layout2 = QHBoxLayout()
        self.test_btn = QPushButton("🎮 테스트 실행")
        self.test_btn.clicked.connect(self.toggle_test)
        btn_layout2.addWidget(self.test_btn)
        
        self.reset_btn = QPushButton("🔄 환경 리셋")
        self.reset_btn.clicked.connect(self.reset_environment)
        btn_layout2.addWidget(self.reset_btn)
        left_panel.addLayout(btn_layout2)
        
        btn_layout3 = QHBoxLayout()
        self.export_btn = QPushButton("📤 아두이노 코드 내보내기")
        self.export_btn.clicked.connect(self.export_arduino)
        btn_layout3.addWidget(self.export_btn)
        
        self.save_btn = QPushButton("💾 저장")
        self.save_btn.clicked.connect(self.save_model)
        btn_layout3.addWidget(self.save_btn)
        
        self.load_btn = QPushButton("📂 로드")
        self.load_btn.clicked.connect(self.load_model)
        btn_layout3.addWidget(self.load_btn)
        left_panel.addLayout(btn_layout3)
        
        # 진행률
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("%v / %m (%p%)")
        left_panel.addWidget(self.progress_bar)
        
        # 상태
        self.status_label = QLabel("대기 중...")
        self.status_label.setStyleSheet("color: #F39C12; font-weight: bold; font-size: 12px;")
        left_panel.addWidget(self.status_label)
        
        main_layout.addLayout(left_panel)
        
        # === 오른쪽 패널 (탭) ===
        tabs = QTabWidget()
        
        # 그래프 탭
        graph_tab = QWidget()
        graph_layout = QVBoxLayout(graph_tab)
        self.graph_widget = GraphWidget()
        graph_layout.addWidget(self.graph_widget)
        tabs.addTab(graph_tab, "학습 그래프")
        
        # 정책 테이블 탭
        policy_tab = QWidget()
        policy_layout = QVBoxLayout(policy_tab)
        
        self.policy_table = QTableWidget()
        self.policy_table.setColumnCount(6)
        self.policy_table.setHorizontalHeaderLabels([
            '상태', '좌센서', '중센서', '우센서', '최적행동', 'Q값'
        ])
        self.policy_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        policy_layout.addWidget(self.policy_table)
        
        refresh_btn = QPushButton("🔄 정책 테이블 새로고침")
        refresh_btn.clicked.connect(self.update_policy_table)
        policy_layout.addWidget(refresh_btn)
        tabs.addTab(policy_tab, "정책 테이블")
        
        # 로그 탭
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        clear_btn = QPushButton("🗑 로그 지우기")
        clear_btn.clicked.connect(lambda: self.log_text.clear())
        log_layout.addWidget(clear_btn)
        tabs.addTab(log_tab, "학습 로그")
        
        main_layout.addWidget(tabs)
        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 1)
    
    def update_env_info_label(self):
        """환경 정보 라벨 업데이트"""
        total = self.obstacle_size_cm + self.road_width_cm * 2
        self.env_info_label.setText(f"전체 환경: {total}x{total}cm")
    
    def on_obstacle_changed(self, value):
        """장애물 크기 변경"""
        self.obstacle_size_cm = value
        self.env.set_obstacle_size(value)
        self.canvas.update_size()
        self.canvas.update()
        self.update_env_info_label()
    
    def on_road_changed(self, value):
        """도로폭 변경"""
        self.road_width_cm = value
        self.env.set_road_width(value)
        self.canvas.update_size()
        self.canvas.update()
        self.update_env_info_label()
    
    def log(self, msg):
        self.log_text.append(msg)
    
    def start_training(self):
        """학습 시작"""
        try:
            self.agent = QLearningAgent(
                learning_rate=self.lr_spin.value(),
                discount_factor=self.gamma_spin.value(),
                epsilon_decay=self.decay_spin.value()
            )
            
            episodes = self.episodes_spin.value()
            visualize = self.visualize_combo.currentIndex() == 1
            
            self.progress_bar.setMaximum(episodes)
            self.progress_bar.setValue(0)
            
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            
            # 환경 설정 비활성화
            self.obstacle_spin.setEnabled(False)
            self.road_spin.setEnabled(False)
            self.obstacle_slider.setEnabled(False)
            self.road_slider.setEnabled(False)
            
            self.log("=" * 40)
            self.log("학습 시작")
            self.log(f"  환경: 장애물 {self.obstacle_size_cm}cm, 도로폭 {self.road_width_cm}cm")
            self.log(f"  학습 횟수: {episodes}")
            self.log(f"  학습률: {self.lr_spin.value()}")
            self.log(f"  할인율: {self.gamma_spin.value()}")
            self.log(f"  감소율: {self.decay_spin.value()}")
            self.log("-" * 40)
            
            self.training_thread = TrainingThread(
                self.agent, episodes,
                self.obstacle_size_cm, self.road_width_cm,
                visualize
            )
            self.training_thread.progress_signal.connect(self.on_progress)
            self.training_thread.finished_signal.connect(self.on_finished)
            self.training_thread.error_signal.connect(self.on_error)
            self.training_thread.update_env_signal.connect(self.on_env_update)
            self.training_thread.start()
            
            self.status_label.setText("학습 진행 중...")
            self.status_label.setStyleSheet("color: #2ECC71; font-weight: bold;")
            
        except Exception as e:
            self.log(f"학습 시작 오류: {e}")
            QMessageBox.critical(self, "오류", f"학습 시작 실패:\n{e}")
            self.start_btn.setEnabled(True)
    
    def on_env_update(self, x, y, angle, steps, laps, checkpoints):
        self.env.car_x = x
        self.env.car_y = y
        self.env.car_angle = angle
        self.env.steps = steps
        self.env.laps_completed = laps
        self.env.checkpoint_passed = checkpoints
        self.canvas.update()
    
    def on_progress(self, episode, reward, epsilon, laps, avg_reward):
        self.progress_bar.setValue(episode)
        
        if episode % 50 == 0:
            self.log(f"에피소드 {episode}: 보상={reward:.1f}, eps={epsilon:.3f}, 완주={laps}")
        
        if episode % 10 == 0:
            self.graph_widget.update_graphs(
                self.agent.episode_rewards,
                self.agent.episode_laps,
                self.agent.epsilon_history
            )
        
        self.status_label.setText(
            f"학습 중... 에피소드 {episode} | 평균보상: {avg_reward:.1f}"
        )
    
    def on_error(self, error_msg):
        self.log(f"오류: {error_msg}")
        QMessageBox.warning(self, "학습 오류", error_msg)
    
    def on_finished(self):
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setText("⏸ 일시정지")
        
        # 환경 설정 활성화
        self.obstacle_spin.setEnabled(True)
        self.road_spin.setEnabled(True)
        self.obstacle_slider.setEnabled(True)
        self.road_slider.setEnabled(True)
        
        if self.agent.episode_laps:
            total_laps = sum(self.agent.episode_laps)
            max_laps = max(self.agent.episode_laps) if self.agent.episode_laps else 0
            self.log("=" * 40)
            self.log("학습 완료!")
            self.log(f"  총 에피소드: {len(self.agent.episode_rewards)}")
            self.log(f"  총 완주: {total_laps}바퀴")
            self.log(f"  최대 완주: {max_laps}바퀴/에피소드")
            
            self.status_label.setText(f"학습 완료! 총 {total_laps}바퀴 완주")
            self.status_label.setStyleSheet("color: #2ECC71; font-weight: bold;")
        
        self.graph_widget.update_graphs(
            self.agent.episode_rewards,
            self.agent.episode_laps,
            self.agent.epsilon_history
        )
        self.update_policy_table()
    
    def toggle_pause(self):
        if self.training_thread:
            self.training_thread.toggle_pause()
            if self.training_thread.paused:
                self.pause_btn.setText("▶ 계속")
                self.status_label.setText("일시정지")
                self.status_label.setStyleSheet("color: #F39C12; font-weight: bold;")
            else:
                self.pause_btn.setText("⏸ 일시정지")
                self.status_label.setText("학습 진행 중...")
                self.status_label.setStyleSheet("color: #2ECC71; font-weight: bold;")
    
    def stop_training(self):
        if self.training_thread:
            self.training_thread.stop()
            self.training_thread.wait(3000)
        
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setText("⏸ 일시정지")
        
        # 환경 설정 활성화
        self.obstacle_spin.setEnabled(True)
        self.road_spin.setEnabled(True)
        self.obstacle_slider.setEnabled(True)
        self.road_slider.setEnabled(True)
        
        self.status_label.setText("학습 중지됨")
        self.status_label.setStyleSheet("color: #E74C3C; font-weight: bold;")
        self.log("학습 중지됨")
    
    def toggle_test(self):
        if self.testing:
            self.test_timer.stop()
            self.testing = False
            self.test_btn.setText("🎮 테스트 실행")
            self.status_label.setText("테스트 종료")
        else:
            self.env.reset()
            self.testing = True
            self.test_btn.setText("⏹ 테스트 중지")
            self.status_label.setText("테스트 실행 중...")
            self.status_label.setStyleSheet("color: #9B59B6; font-weight: bold;")
            self.test_timer.start(80)
    
    def test_step(self):
        if self.env.done:
            if self.env.collision:
                self.status_label.setText(f"충돌! 완주: {self.env.laps_completed}바퀴")
                self.status_label.setStyleSheet("color: #E74C3C; font-weight: bold;")
            else:
                self.status_label.setText(f"시간초과! 완주: {self.env.laps_completed}바퀴")
            self.test_timer.stop()
            self.testing = False
            self.test_btn.setText("🎮 테스트 실행")
            return
        
        state = self.env.get_state()
        action = self.agent.choose_action(state, training=False)
        self.env.step(action)
        self.canvas.update()
        
        self.status_label.setText(
            f"테스트 중... 스텝: {self.env.steps} | 완주: {self.env.laps_completed}바퀴"
        )
    
    def reset_environment(self):
        self.env.reset()
        self.canvas.update()
        self.status_label.setText("환경 리셋됨")
    
    def update_policy_table(self):
        try:
            policy = self.agent.get_policy_table()
            self.policy_table.setRowCount(len(policy))
            
            action_names = ['전진', '좌회전', '우회전', '급좌회전', '급우회전']
            dist_names = ['위험', '가까움', '보통', '멀음', '안전']
            
            row = 0
            for state, (action, q_val) in sorted(policy.items()):
                state_idx = state[0] * 25 + state[1] * 5 + state[2]
                
                item = QTableWidgetItem(str(state_idx))
                item.setTextAlignment(Qt.AlignCenter)
                self.policy_table.setItem(row, 0, item)
                
                for i, val in enumerate(state):
                    item = QTableWidgetItem(dist_names[val])
                    item.setTextAlignment(Qt.AlignCenter)
                    if val == 0:
                        item.setBackground(QColor(231, 76, 60, 100))
                    elif val == 4:
                        item.setBackground(QColor(46, 204, 113, 100))
                    self.policy_table.setItem(row, i + 1, item)
                
                item = QTableWidgetItem(action_names[action])
                item.setTextAlignment(Qt.AlignCenter)
                colors = [
                    QColor(46, 204, 113, 100),
                    QColor(241, 196, 15, 100),
                    QColor(155, 89, 182, 100),
                    QColor(230, 126, 34, 100),
                    QColor(52, 152, 219, 100),
                ]
                item.setBackground(colors[action])
                self.policy_table.setItem(row, 4, item)
                
                item = QTableWidgetItem(f"{q_val:.2f}")
                item.setTextAlignment(Qt.AlignCenter)
                self.policy_table.setItem(row, 5, item)
                
                row += 1
        except Exception as e:
            self.log(f"정책 테이블 업데이트 오류: {e}")
    
    def export_arduino(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "아두이노 코드 저장", "circuit_car.ino", "Arduino (*.ino)"
        )
        if filename:
            try:
                self.generate_arduino_code(filename)
                QMessageBox.information(self, "완료", f"아두이노 코드 저장됨:\n{filename}")
                self.log(f"아두이노 코드 저장: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "오류", f"저장 실패:\n{e}")
    
    def generate_arduino_code(self, filename):
        policy = self.agent.get_policy_table()
        
        policy_array = []
        for i in range(125):
            left = i // 25
            center = (i % 25) // 5
            right = i % 5
            action, _ = policy.get((left, center, right), (0, 0))
            policy_array.append(action)
        
        total_size = self.obstacle_size_cm + self.road_width_cm * 2
        
        code = f'''/*
 * Circuit AI Car - Reinforcement Learning
 * 
 * Environment:
 * - Total size: {total_size}x{total_size}cm
 * - Obstacle: {self.obstacle_size_cm}x{self.obstacle_size_cm}cm
 * - Road width: {self.road_width_cm}cm
 * 
 * Pumpkin Factory Makerspace
 */

#include <avr/pgmspace.h>

#define TRIG_LEFT 2
#define ECHO_LEFT 3
#define TRIG_CENTER 4
#define ECHO_CENTER 5
#define TRIG_RIGHT 6
#define ECHO_RIGHT 7

#define ENA 9
#define IN1 8
#define IN2 10
#define ENB 11
#define IN3 12
#define IN4 13

#define MOTOR_SPEED 180
#define TURN_SPEED 160
#define SHARP_TURN_SPEED 200
#define SENSOR_TIMEOUT 30000

#define ACTION_FORWARD 0
#define ACTION_LEFT 1
#define ACTION_RIGHT 2
#define ACTION_SHARP_LEFT 3
#define ACTION_SHARP_RIGHT 4

const uint8_t POLICY[125] PROGMEM = {{
    {', '.join(map(str, policy_array))}
}};

long measureDistance(int trigPin, int echoPin) {{
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);
    
    long duration = pulseIn(echoPin, HIGH, SENSOR_TIMEOUT);
    if (duration == 0) return 100;
    return constrain(duration * 0.034 / 2, 0, 100);
}}

int discretize(long dist) {{
    if (dist < 10) return 0;
    else if (dist < 20) return 1;
    else if (dist < 30) return 2;
    else if (dist < 40) return 3;
    else return 4;
}}

int getAction(int left, int center, int right) {{
    int idx = left * 25 + center * 5 + right;
    if (idx < 0 || idx >= 125) return ACTION_FORWARD;
    return pgm_read_byte(&POLICY[idx]);
}}

void forward() {{
    digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
    digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
    analogWrite(ENA, MOTOR_SPEED);
    analogWrite(ENB, MOTOR_SPEED);
}}

void turnLeft() {{
    digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
    digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
    analogWrite(ENA, 0);
    analogWrite(ENB, TURN_SPEED);
}}

void turnRight() {{
    digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
    digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
    analogWrite(ENA, TURN_SPEED);
    analogWrite(ENB, 0);
}}

void sharpLeft() {{
    digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
    digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
    analogWrite(ENA, SHARP_TURN_SPEED);
    analogWrite(ENB, SHARP_TURN_SPEED);
}}

void sharpRight() {{
    digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
    digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
    analogWrite(ENA, SHARP_TURN_SPEED);
    analogWrite(ENB, SHARP_TURN_SPEED);
}}

void stopMotors() {{
    digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
    digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
    analogWrite(ENA, 0); analogWrite(ENB, 0);
}}

void setup() {{
    Serial.begin(9600);
    Serial.println("Circuit AI Car - {total_size}x{total_size}cm");
    
    pinMode(TRIG_LEFT, OUTPUT); pinMode(ECHO_LEFT, INPUT);
    pinMode(TRIG_CENTER, OUTPUT); pinMode(ECHO_CENTER, INPUT);
    pinMode(TRIG_RIGHT, OUTPUT); pinMode(ECHO_RIGHT, INPUT);
    
    pinMode(ENA, OUTPUT); pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
    pinMode(ENB, OUTPUT); pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
    
    stopMotors();
    delay(2000);
}}

void loop() {{
    long dL = measureDistance(TRIG_LEFT, ECHO_LEFT);
    long dC = measureDistance(TRIG_CENTER, ECHO_CENTER);
    long dR = measureDistance(TRIG_RIGHT, ECHO_RIGHT);
    
    int sL = discretize(dL);
    int sC = discretize(dC);
    int sR = discretize(dR);
    
    if (dC < 5 || dL < 5 || dR < 5) {{
        stopMotors();
        delay(100);
        digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
        digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
        analogWrite(ENA, MOTOR_SPEED);
        analogWrite(ENB, MOTOR_SPEED);
        delay(300);
        stopMotors();
        return;
    }}
    
    int action = getAction(sL, sC, sR);
    
    switch (action) {{
        case ACTION_FORWARD: forward(); break;
        case ACTION_LEFT: turnLeft(); break;
        case ACTION_RIGHT: turnRight(); break;
        case ACTION_SHARP_LEFT: sharpLeft(); break;
        case ACTION_SHARP_RIGHT: sharpRight(); break;
        default: forward();
    }}
    
    static unsigned long lastPrint = 0;
    if (millis() - lastPrint > 500) {{
        Serial.print("D:"); Serial.print(dL);
        Serial.print("/"); Serial.print(dC);
        Serial.print("/"); Serial.print(dR);
        Serial.print(" S:"); Serial.print(sL);
        Serial.print(","); Serial.print(sC);
        Serial.print(","); Serial.print(sR);
        Serial.print(" A:"); Serial.println(action);
        lastPrint = millis();
    }}
    
    delay(50);
}}
'''
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(code)
    
    def save_model(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "모델 저장", "circuit_model.json", "JSON (*.json)"
        )
        if filename:
            try:
                self.agent.save_model(filename)
                QMessageBox.information(self, "완료", "모델이 저장되었습니다.")
                self.log(f"모델 저장: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "오류", f"저장 실패:\n{e}")
    
    def load_model(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "모델 로드", "", "JSON (*.json)"
        )
        if filename:
            try:
                self.agent.load_model(filename)
                self.graph_widget.update_graphs(
                    self.agent.episode_rewards,
                    self.agent.episode_laps,
                    self.agent.epsilon_history
                )
                self.update_policy_table()
                QMessageBox.information(self, "완료", "모델이 로드되었습니다.")
                self.log(f"모델 로드: {filename}")
            except Exception as e:
                QMessageBox.warning(self, "오류", f"로드 실패:\n{e}")
    
    def closeEvent(self, event):
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop()
            self.training_thread.wait(2000)
        event.accept()


# ============================================================
# 메인
# ============================================================
def main():
    try:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        
        # 한글 폰트 설정
        font = QFont('Malgun Gothic', 9)
        app.setFont(font)
        
        window = MainWindow()
        window.show()
        
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Application error: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()