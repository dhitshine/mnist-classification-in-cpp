add_executable(app main.cpp)

target_link_libraries(app PRIVATE header)

add_custom_command(TARGET app POST_BUILD      # bikin symlink folder mnist
  COMMAND ${CMAKE_COMMAND} -E create_symlink
  "${CMAKE_SOURCE_DIR}/mnist"
  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mnist"
  COMMENT "Creating symlink for MNIST"
)
