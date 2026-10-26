# Task 3B - Push Together

## What to do

Three robots, `r1`, `r2` and `r3`, move **one payload** together. Each robot docks on its own side of `payload_gamma`, welds it to itself with the `/rN/attach_payload` service, and the three carry it as one rigid body onto the **hexagon's centre**, then let go with `/rN/detach_payload` and back away. Nothing may touch a pillar, a station, the arena wall, another payload or another robot.

The fallen pillars move to a **new random layout every launch**, so plan from the camera, never from a fixed route. **Where the payload is, you find yourself**, in the camera: it is not given anywhere. Its destination and each robot's side of it are given in the code:

| | Where (`/odom` frame, m) |
| --- | --- |
| `payload_gamma` | its top is violet (lapis): find it in the camera |
| Destination: the hexagon's centre | `(1.2192, 1.2192)` |
| `r1` docks | above it, facing down |
| `r2` docks | left of it, facing right |
| `r3` docks | below it, facing up |

**Attaching.** `/rN/attach_payload` (`hb_description/srv/AttachPayload`, `payload_name: payload_gamma`) is refused unless the robot faces the payload (within ±20° of its red arrow) and the tip of its **solenoid** is right up against it. The reply's `message` says which rule failed. Once all three are attached, the four bodies are one: if every robot keeps its offset from the payload and its heading, the shape holds.

It's correct when the planner logs `published /carry_plan`, the follower logs `all three attached`, then `all three let go`, then `task 3B complete`, and the payload rests on the hexagon's centre.

## Files to edit

```
Software/workspace/src/task_3b/task_3b/path_planner.py
Software/workspace/src/task_3b/task_3b/path_follower.py
```

Fill in the `ADD YOUR CODE HERE` blocks:

1. `path_planner.py`: write `find_payload()`, which finds `payload_gamma` in the camera frame (as Task 1A found the checkpoints); copy in `remove_drivable()` and `to_path_msg()` (Task 2B) and `plan_all()` (Task 2C, now with a grid and a goal per robot); then write `plan_carry()`: the payload's centre from where you found it to the hexagon on the carry grid.
2. `path_follower.py`: paste your Task 2A code into **FROM TASK 2A**, copy in `path_to_traj()` (Task 2B), then write `request_attach()` and `request_detach()`: send the request with `call_async()` and handle the reply in a done-callback, without blocking `tick()`.

Leave the camera, grid, node and team code as given.

## Run

```bash
cd ~/eYRC_26-27_Hola-The-Explorer
git pull
cd Software/workspace
colcon build
source install/setup.bash
```

```bash
# Terminal 1 - simulation (new obstacle layout every launch)
ros2 launch hb_description task3b.launch.py

# Terminal 2 - planner
ros2 run task_3b path_planner

# Terminal 3 - follower
ros2 run task_3b path_follower
```

Try a service call by hand first:

```bash
ros2 service call /r2/attach_payload hb_description/srv/AttachPayload "{payload_name: payload_gamma}"
```

Relaunch a few times: a run that only works on one layout is not done.
