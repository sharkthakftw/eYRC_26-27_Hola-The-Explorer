# Copyright (c) 2026 e-Yantra, IIT Bombay. All rights reserved.
# These simulation files and source code are the intellectual property of e-Yantra,
# IIT Bombay, provided solely for eYRC 2026-27 (Theme: Hola The Explorer).
# Sharing or redistribution of this material, in whole or in part, is not permitted.

cmake_minimum_required(VERSION 3.8)
project(task_3b)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  task_3b/path_planner.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME path_planner
)

install(PROGRAMS
  task_3b/path_follower.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME path_follower
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
