#!/bin/bash

module load nvhpc/24.5
cmake -B build 
cmake --build build -j