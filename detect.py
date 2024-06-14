import jetson_inference
import jetson_utils

import argparse
import sys


def parse_args():
	parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.")
	parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
	parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
	parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
	parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
	parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

	try:
		opt = parser.parse_known_args()[0]
		return opt
	except:
		print("")
		parser.print_help()
		sys.exit(0)


if __name__ == "__main__":
	opt = parse_args()
	
	# load the object detection network
	net = jetson_inference.detectNet(opt.network, sys.argv, opt.threshold)
	net.SetTrackingEnabled(True)
	net.SetTrackingParams(minFrames=3, dropFrames=15, overlapThreshold=0.5)

	# create video sources & outputs
	input = jetson_utils.videoSource(opt.input_URI, argv=sys.argv)
	output = jetson_utils.videoOutput(opt.output_URI, argv=sys.argv)

	# process frames until the user exits
	while True:
		# capture the next image
		img = input.Capture()
		if img is None:
			continue

		# detect objects in the image (with overlay)
		detections = net.Detect(img, overlay=opt.overlay)

		# print the detections
		print("detected {:d} objects in image".format(len(detections)))
		for detection in detections:
			if detection.TrackStatus >= 0:  # actively tracking
				print(f"object {detection.TrackID} at ({detection.Left}, {detection.Top}) has been tracked for {detection.TrackFrames} frames")
			else:  # if tracking was lost, this object will be dropped the next frame
				print(f"object {detection.TrackID} has lost tracking")   

		# render the image
		output.Render(img)

		# update the title bar
		output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

		# print out performance info
		net.PrintProfilerTimes()

		# exit on input/output EOS
		if not input.IsStreaming() or not output.IsStreaming():
			break
