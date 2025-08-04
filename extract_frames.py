#!/usr/bin/env python3
"""
Magic Leap Camera Frame Extractor

Extracts individual JPEG frames from MLCF (Magic Leap Camera Frames) container files.
The container format is:
- Magic bytes (4 bytes): "MLCF"
- Frame count (4 bytes): Total number of frames
- First frame offset (4 bytes): Start position of frame data
- Frame data: [frame_size:4bytes][frame_id_length:1byte][frame_id:string][jpeg_data] repeated

Usage:
    python extract_frames.py input.mlcf output_folder/
"""

import os
import struct
import sys
from pathlib import Path


def read_container_header(file):
    """Read and validate the MLCF container header."""
    magic = file.read(4)
    if magic != b'MLCF':
        raise ValueError(f"Invalid magic bytes: expected b'MLCF', got {magic}")
    
    frame_count = struct.unpack('<I', file.read(4))[0]
    first_frame_offset = struct.unpack('<I', file.read(4))[0]
    
    return frame_count, first_frame_offset


def extract_frame(file):
    """Extract a single frame from the current file position."""
    # Read frame size
    frame_size_data = file.read(4)
    if len(frame_size_data) < 4:
        return None  # End of file
    
    frame_size = struct.unpack('<I', frame_size_data)[0]
    
    # Read frame ID length
    frame_id_length = struct.unpack('<B', file.read(1))[0]
    
    # Read frame ID
    frame_id = file.read(frame_id_length).decode('utf-8')
    
    # Calculate JPEG data size: frame_size - 1 byte (ID length) - ID bytes
    jpeg_size = frame_size - 1 - frame_id_length
    
    # Read JPEG data
    jpeg_data = file.read(jpeg_size)
    
    return frame_id, jpeg_data


def extract_frames(input_file, output_dir):
    """Extract all frames from MLCF container to individual JPEG files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(input_file, 'rb') as f:
        try:
            frame_count, first_frame_offset = read_container_header(f)
            
            # Handle incomplete recordings where frame count is 0
            if frame_count == 0:
                print("Warning: Frame count is 0 - recording may not have been properly stopped")
                print("Attempting to recover frames by scanning file...")
            else:
                print(f"Container has {frame_count} frames")
            
            # Seek to first frame
            f.seek(first_frame_offset)
            
            extracted_count = 0
            while True:
                try:
                    frame_data = extract_frame(f)
                    if frame_data is None:
                        break
                    
                    frame_id, jpeg_data = frame_data
                    
                    # Validate JPEG data (should start with FF D8 and end with FF D9)
                    if len(jpeg_data) < 4 or jpeg_data[:2] != b'\xff\xd8':
                        print(f"Warning: Frame {frame_id} doesn't appear to be valid JPEG data, skipping...")
                        continue
                    
                    output_file = output_dir / f"{frame_id}.jpg"
                    
                    with open(output_file, 'wb') as out_f:
                        out_f.write(jpeg_data)
                    
                    extracted_count += 1
                    if extracted_count % 100 == 0:
                        print(f"Extracted {extracted_count} frames...")
                        
                except struct.error as e:
                    print(f"Reached end of valid data or corrupted frame at position {f.tell()}")
                    break
                except UnicodeDecodeError as e:
                    print(f"Invalid frame ID encoding at position {f.tell()}, stopping extraction")
                    break
                except Exception as e:
                    print(f"Error extracting frame at position {f.tell()}: {e}")
                    # Try to continue with next frame
                    continue
            
            print(f"Successfully extracted {extracted_count} frames to {output_dir}")
            
            if frame_count > 0 and extracted_count != frame_count:
                print(f"Warning: Header claimed {frame_count} frames, but extracted {extracted_count}")
            elif frame_count == 0 and extracted_count > 0:
                print(f"Successfully recovered {extracted_count} frames from incomplete recording!")
                
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return False
    
    return True


def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_frames.py input.mlcf output_folder/")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found")
        sys.exit(1)
    
    print(f"Extracting frames from {input_file} to {output_dir}")
    
    if extract_frames(input_file, output_dir):
        print("Extraction completed successfully!")
    else:
        print("Extraction failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()