import numpy as np
import os

import game
import dqn

import importlib
importlib.reload(game)
importlib.reload(dqn)

model = 130
folder = 'models'

record_video = False

env = game.Environment()

agent = dqn.DQN(
    state_shape=env.ENVIRONMENT_SHAPE,
    action_size=env.ACTION_SPACE_SIZE
)

agent.load(f'{folder}/{model}.h5')

state = env.reset()
state = np.expand_dims(state, axis=0)

import pygame
pygame.init()
screen = pygame.display.set_mode((env.WINDOW_WIDTH, env.WINDOW_HEIGHT))
clock = pygame.time.Clock()
running = True
score = 0

import record
recorder = None
if record_video:
    recorder = record.ScreenRecorder(env.WINDOW_WIDTH, env.WINDOW_HEIGHT, env.FPS, f"{folder}_{model}.avi")

while running:

    pygame.display.set_caption(f"Score: {score}")

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = agent.act(state, 0)
    state, reward, done, score = env.step(action)
    state = np.expand_dims(state, axis=0)

    env.render(screen)
    pygame.display.flip()
    clock.tick(15)

    if record_video:
        recorder.capture_frame(screen)

pygame.quit()
if record_video:
    recorder.end_recording()