import cv2
import pygame
import math
import sys
import os
import numpy as np
from stl import mesh
from collections import defaultdict

# ----------------------------
# STL Model: Load + Normalize + Extract Edges
# ----------------------------
class STLModel:
    def __init__(self, filepath, target_size=2.0):
        self.mesh = mesh.Mesh.from_file(filepath)
        self.edges = self._normalize_and_extract_edges(target_size)

    def _normalize_and_extract_edges(self, target_size):
        all_vertices = self.mesh.vectors.reshape(-1, 3)
        center = np.mean(all_vertices, axis=0)
        all_vertices -= center
        max_dim = np.max(np.linalg.norm(all_vertices, axis=1))
        all_vertices *= (target_size / max_dim)
        self.mesh.vectors = all_vertices.reshape(-1, 3, 3)

        edge_map = defaultdict(list)
        for tri in self.mesh.vectors:
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            norm = np.linalg.norm(normal)
            normal = normal / norm if norm else normal
            for i in range(3):
                p1 = tuple(tri[i])
                p2 = tuple(tri[(i + 1) % 3])
                edge = tuple(sorted((p1, p2)))
                edge_map[edge].append(tuple(normal))

        visible_edges = []
        for edge, normals in edge_map.items():
            if len(normals) == 1:
                visible_edges.append(edge)
            elif len(normals) == 2 and np.dot(normals[0], normals[1]) < 0.999:
                visible_edges.append(edge)
        return visible_edges

# ----------------------------
# Renderer: Rotate, Project, Draw
# ----------------------------
class WireframeRenderer:
    def __init__(self, width, height, scale=200, distance=5):
        self.width = width
        self.height = height
        self.scale = scale
        self.distance = distance

    def rotate(self, p, angle_x, angle_y):
        x, y, z = p
        sin_x, cos_x = math.sin(angle_x), math.cos(angle_x)
        sin_y, cos_y = math.sin(angle_y), math.cos(angle_y)
        y, z = y * cos_x - z * sin_x, y * sin_x + z * cos_x
        x, z = x * cos_y + z * sin_y, -x * sin_y + z * cos_y
        return [x, y, z]

    def project(self, p):
        x, y, z = p
        factor = self.scale / (z + self.distance)
        return int(self.width / 2 + x * factor), int(self.height / 2 - y * factor)

    def draw(self, screen, edges, angle_x, angle_y):
        screen.fill((0, 0, 0))  # true black background

        # Glowing pulse effect
        t = pygame.time.get_ticks() * 0.005
        glow_intensity = 180 + int(50 * math.sin(t))
        glow_color = (0, glow_intensity, 180)
        core_color = (0, 255, 255)

        # Chromatic aberration offset
        offset = 1

        for p1, p2 in edges:
            p1_rot = self.rotate(p1, angle_x, angle_y)
            p2_rot = self.rotate(p2, angle_x, angle_y)
            p1_proj = self.project(p1_rot)
            p2_proj = self.project(p2_rot)

            # Aberration: red-blue shadow lines slightly offset
            pygame.draw.line(screen, (0, 80, 255), (p1_proj[0] - offset, p1_proj[1]), (p2_proj[0] - offset, p2_proj[1]), 2)
            pygame.draw.line(screen, (0, 255, 80), (p1_proj[0] + offset, p1_proj[1]), (p2_proj[0] + offset, p2_proj[1]), 2)

            # Glow layers
            pygame.draw.line(screen, glow_color, p1_proj, p2_proj, 5)
            pygame.draw.line(screen, glow_color, p1_proj, p2_proj, 3)

            # Core line
            pygame.draw.line(screen, core_color, p1_proj, p2_proj, 1)

        # Animated scanlines
        scan_offset = int((pygame.time.get_ticks() * 0.1) % 4)
        for y in range(scan_offset, self.height, 4):
            pygame.draw.line(screen, (0, 40, 40), (0, y), (self.width, y), 1)


# ----------------------------
# Head Tracker: OpenCV face tracking
# ----------------------------
class HeadTracker:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.cap = cv2.VideoCapture(0)
        self.last_angle_x = 0
        self.last_angle_y = 0
        self.MIN_FACE_SIZE = 100
        self.REAL_FACE_WIDTH = 16.0
        self.FOCAL_LENGTH = 500.0
        self.MAX_ANGLE = math.pi / 2
        self.NEAR_DIST = 30
        self.FAR_DIST = 80

    def get_angles(self):
        ret, frame = self.cap.read()
        if not ret:
            return self.last_angle_x, self.last_angle_y, frame, False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        faces = [f for f in faces if f[2] > self.MIN_FACE_SIZE and f[3] > self.MIN_FACE_SIZE]

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cx, cy = x + w / 2, y + h / 2
            distance_cm = (self.REAL_FACE_WIDTH * self.FOCAL_LENGTH) / w
            offset_x = (cx / frame.shape[1]) - 0.5
            offset_y = 0.5 - (cy / frame.shape[0])
            scale = (max(self.NEAR_DIST, min(self.FAR_DIST, distance_cm)) - self.NEAR_DIST) / (self.FAR_DIST - self.NEAR_DIST)
            max_angle = (1 - scale) * self.MAX_ANGLE

            angle_y = -offset_x * max_angle
            angle_x = -offset_y * max_angle
            self.last_angle_x, self.last_angle_y = angle_x, angle_y
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return angle_x, angle_y, frame, True

        return self.last_angle_x, self.last_angle_y, frame, False

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

# ----------------------------
# App: Runs everything
# ----------------------------
class App:
    def __init__(self):
        self.WIDTH, self.HEIGHT = 640, 480
        pygame.display.set_caption("Holographic Wireframe Viewer")
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "cube.stl")
        if not os.path.isfile(filepath):
            print("❌ STL file not found:", filepath)
            sys.exit(1)

        self.model = STLModel(filepath)
        self.renderer = WireframeRenderer(self.WIDTH, self.HEIGHT)
        self.tracker = HeadTracker()

    def run(self):
        running = True
        angle_x = 0
        angle_y = 0
        face_visible_prev = False
        slow_smoothing_frames = 0  # counter for how long to apply slow smoothing

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            target_angle_x, target_angle_y, frame, face_visible = self.tracker.get_angles()

            # If face was just reacquired, enter slow smoothing mode for a few frames
            if face_visible and not face_visible_prev:
                slow_smoothing_frames = 15  # adjust duration of glide here

            if face_visible:
                if slow_smoothing_frames > 0:
                    smoothing = 0.05  # glide toward new position
                    slow_smoothing_frames -= 1
                else:
                    smoothing = 0.5  # fast tracking once aligned
            else:
                smoothing = 0.05  # slowly glide to last known target

            angle_x += smoothing * (target_angle_x - angle_x)
            angle_y += smoothing * (target_angle_y - angle_y)

            face_visible_prev = face_visible

            cv2.imshow("Face Tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            self.renderer.draw(self.screen, self.model.edges, angle_x, angle_y)
            pygame.display.flip()
            self.clock.tick(30)

        self.tracker.release()
        pygame.quit()
        sys.exit()



# ----------------------------
# Run it
# ----------------------------
if __name__ == "__main__":
    App().run()
