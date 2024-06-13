import argparse

# Initialize the parser
test_parser = argparse.ArgumentParser(description="Test Script Configuration")

test_parser.add_argument('--dataroot', default='./horse2zebra', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)') 
test_parser.add_argument('--path_testA', type=str, required=True, deafult = 'test_A', help='Path to testing dataset A')
test_parser.add_argument('--path_testB', type=str, required=True, default = 'test_B', help='Path to testing dataset B')
test_parser.add_argument('--num_threads', type=int, default=0, help='Number of threads, fixed at 0 for this test script')
test_parser.add_argument('--batch_size', type=int, default=1, help='Batch size, fixed at 1 for this test script')
test_parser.add_argument('--serial_batches', action='store_true', default=True, help='Disable data shuffling, fixed as True')
test_parser.add_argument('--no_flip', action='store_true', default=True, help='No flip, fixed as True')
test_parser.add_argument('--aspect_ratio', type=float, default=1.0, help='Aspect ratio for the test')
test_parser.add_argument('--create_dir', type=str, default='/kaggle/working/test_dir', help='Directory to create for output images')