#!/bin/bash

gst-launch-1.0 ximagesrc use-damage=0 ! video/x-raw ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! nvv4l2h264enc ! h264parse ! matroskamux ! filesink location=record.mkv
