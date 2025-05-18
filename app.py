import cv2
import pygame
import math
import sys
from stl import mesh
import numpy as np

# --- Pygame setup ---
pygame.init()
WIDTH, HEIGHT = 640, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

import os
from stl import mesh

# Get path to current script
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "models", "ant.stl")

# Check if file exists
if not os.path.isfile(file_path):
    print("❌ File not found:", file_path)
    exit(1)

# Load STL
your_mesh = mesh.Mesh.from_file(file_path)


# Extract unique edges from the triangles
def extract_edges(stl_mesh):
    edges = set()
    for triangle in stl_mesh.vectors:
        for i in range(3):
            p1 = tuple(triangle[i])
            p2 = tuple(triangle[(i + 1) % 3])
            edge = tuple(sorted([p1, p2]))
            edges.add(edge)
    return list(edges)

model_edges = extract_edges(your_mesh)

# --- Transformations ---
def rotate_x(p, angle):
    x, y, z = p
    sin_a = math.sin(angle)
    cos_a = math.cos(angle)
    return [
        x,
        y * cos_a - z * sin_a,
        y * sin_a + z * cos_a
    ]

def rotate_y(p, angle):
    x, y, z = p
    sin_a = math.sin(angle)
    cos_a = math.cos(angle)
    return [
        x * cos_a + z * sin_a,
        y,
        -x * sin_a + z * cos_a
    ]

def project(p):
    scale = 200
    distance = 10
    x, y, z = p
    factor = scale / (z + distance)
    return int(WIDTH / 2 + x * factor), int(HEIGHT / 2 - y * factor)

# --- OpenCV face tracking ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

# Parameters for angle computation
REAL_FACE_WIDTH = 16.0  # cm
FOCAL_LENGTH = 500.0
MAX_ANGLE_AT_NEAR = math.pi / 4
NEAR_DIST = 30
FAR_DIST = 80

def compute_scaled_angle(offset, dist_cm):
    dist_cm = max(NEAR_DIST, min(FAR_DIST, dist_cm))
    scale = 1 - (dist_cm - NEAR_DIST) / (FAR_DIST - NEAR_DIST)
    max_angle = scale * MAX_ANGLE_AT_NEAR
    return offset * max_angle

# Initial state
angle_x = 0
angle_y = 0
last_angle_x = 0
last_angle_y = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            cv2.destroyAllWindows()
            pygame.quit()
            sys.exit()

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    face_found = False
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_center_x = x + w / 2
        face_center_y = y + h / 2
        distance_cm = (REAL_FACE_WIDTH * FOCAL_LENGTH) / w
        offset_x = (face_center_x / frame.shape[1]) - 0.5
        offset_y = 0.5 - (face_center_y / frame.shape[0])
        angle_y = compute_scaled_angle(-offset_x, distance_cm)
        angle_x = compute_scaled_angle(-offset_y, distance_cm)
        last_angle_x = angle_x
        last_angle_y = angle_y
        face_found = True
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        # Use last known angles if no face is detected
        angle_x = last_angle_x
        angle_y = last_angle_y

    # Show webcam frame
    cv2.imshow("Face Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # --- Draw model ---
    screen.fill((30, 30, 30))
    for edge in model_edges:
        p1 = rotate_y(rotate_x(edge[0], angle_x), angle_y)
        p2 = rotate_y(rotate_x(edge[1], angle_x), angle_y)
        proj1 = project(p1)
        proj2 = project(p2)
        pygame.draw.line(screen, (200, 200, 200), proj1, proj2, 1)

    pygame.display.flip()
    clock.tick(30)

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.quit()
