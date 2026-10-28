# Task 3A - Deliver the Payload

## What to do

One robot, `r2`, moves **one payload**, `payload_beta`. It docks on the payload's right side, welds it to itself with the `/r2/attach_payload` service, carries it onto the **hexagon's centre**, lets it go with `/r2/detach_payload`, and drives back to its start circle. Nothing may touch a pillar, a station, the arena wall or another payload.

The fallen pillars move to a **new random layout every launch**, so plan from the camera, never from a fixed route. **Where the payload is, you find yourself**, in the camera: it is not given anywhere. Its destination and the robot's side of it are given in the code:

| | Where (`/odom` frame, m) |
| --- | --- |
| `payload_beta` | its top is teal (verdigris): find it in the camera |
| Destination: the hexagon's centre | `(1.2192, 1.2192)` |
| `r2` docks | right of it, facing left |
| `r2` returns to | its start circle `(1.2192, 2.2616)` |

**Attaching.** `/r2/attach_payload` (`hb_description/srv/AttachPayload`, `payload_name: payload_beta`) is refused unless the robot faces the payload (within ±20° of its red arrow) and the tip of its **solenoid** is right up against it. The reply's `message` says which rule failed. Once attached, the payload moves with the robot as if welded on: if the robot holds its heading, the payload keeps its place in front of the arrow.

It's correct when the planner logs `published /r2/plan`, `published /carry_plan` and `published /r2/return_plan`, the follower logs `attached payload_beta: carrying`, then `let go of payload_beta`, then `task 3A complete`, and the payload rests on the hexagon's centre.

## Files to edit

```
Software/workspace/src/task_3a/task_3a/path_planner.py
Software/workspace/src/task_3a/task_3a/path_follower.py
```

Fill in the `ADD YOUR CODE HERE` blocks:

1. `path_planner.py`: write `find_payload()`, which finds `payload_beta` in the camera frame (as Task 1A found the checkpoints); copy in `remove_drivable()`, `plan_route()` and `to_path_msg()` (Task 2B; `plan_route()` now plans one leg, from a start to a goal); then write `plan_carry()`: the payload's centre from where you found it to the hexagon on the carry grid.
2. `path_follower.py`: paste your Task 2A code into **FROM TASK 2A**, copy in `path_to_traj()` (Task 2B), then write `request_attach()` and `request_detach()`: send the request with `call_async()` and handle the reply in a done-callback, without blocking `tick()`.

Leave the camera, grid and node code as given.

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
ros2 launch hb_description task3a.launch.py

# Terminal 2 - planner
ros2 run task_3a path_planner

# Terminal 3 - follower
ros2 run task_3a path_follower
```

Try a service call by hand first:

```bash
ros2 service call /r2/attach_payload hb_description/srv/AttachPayload "{payload_name: payload_beta}"
```

Relaunch a few times: a run that only works on one layout is not done.
