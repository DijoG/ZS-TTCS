# zs_ttcs/cli.py
#!/usr/bin/env python3
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=['1', '2', '3', 'all'])
    parser.add_argument('input')
    parser.add_argument('--output', '-o', default='./output')
    
    args = parser.parse_args()
    
    if args.step == '1':
        from .step1_segformer_first import main as step1_main
        sys.argv = [sys.argv[0], args.input, '--output', args.output]
        step1_main()
    elif args.step == '2':
        from .step2_mask_ndvi import main as step2_main
        sys.argv = [sys.argv[0], args.input, '--output', args.output]
        step2_main()
    elif args.step == '3':
        from .step3_forman_gradient import main as step3_main
        sys.argv = [sys.argv[0], args.input, '--output', args.output]
        step3_main()
    elif args.step == 'all':
        from .pipeline import run_pipeline
        run_pipeline(args.input, args.output)

if __name__ == '__main__':
    main()