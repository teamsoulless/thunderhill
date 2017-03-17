from moviepy.editor import ImageSequenceClip
import argparse
import cv2
import numpy as np
import utils


utils.setGlobals()
color_space = utils.getGlobals()[0]


def main():
    parser = argparse.ArgumentParser(description='Create driving video.')
    parser.add_argument(
        'image_folder',
        type=str,
        default='',
        help='Path to image folder. The video will be created from these images.'
      )
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='FPS (Frames per second) setting for the video.'
      )
    parser.add_argument(
        '--processed',
        action='store_true',
        help='Transform the image back to the model space.'
      )
    args = parser.parse_args()

    video_file = args.image_folder + '.mp4'
    print("Creating video {}, FPS={}".format(video_file, args.fps))
    clip = ImageSequenceClip(args.image_folder, fps=args.fps)

    if args.processed:
        def tmp(im):
            processed = utils.process_image(im)
            processed = cv2.cvtColor(processed, color_space['src'])
            return np.hstack((im, processed))
        clip = clip.fl_image(tmp)

    clip.write_videofile(video_file)


if __name__ == '__main__':
    main()
