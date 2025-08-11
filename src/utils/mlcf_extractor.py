"""MLCF (Magic Leap Camera Frames) container extractor utility."""

import struct
from pathlib import Path
from typing import Optional, Tuple


class MLCFExtractor:
    """Extracts JPEG frames from MLCF container files.
    
    The container format is:
    - Magic bytes (4 bytes): "MLCF"
    - Frame count (4 bytes): Total number of frames
    - First frame offset (4 bytes): Start position of frame data
    - Frame data: [frame_size:4bytes][frame_id_length:1byte][frame_id:string][jpeg_data] repeated
    """
    
    def __init__(self, verbose: bool = True):
        """Initialize the MLCF extractor.
        
        Args:
            verbose: Whether to print extraction progress
        """
        self.verbose = verbose
    
    def extract(self, mlcf_path: Path, output_dir: Path) -> bool:
        """Extract all frames from MLCF container to individual JPEG files.
        
        Args:
            mlcf_path: Path to the MLCF container file
            output_dir: Directory to save extracted frames
            
        Returns:
            True if extraction was successful, False otherwise
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print(f"Extracting frames from {mlcf_path.name} to {output_dir.name}/")
        
        with open(mlcf_path, 'rb') as f:
            try:
                frame_count, first_frame_offset = self._read_header(f)
                
                if frame_count == 0 and self.verbose:
                    print("  Warning: Frame count is 0 - recording may be incomplete")
                    print("  Attempting to recover frames...")
                elif self.verbose:
                    print(f"  Container has {frame_count} frames")
                
                f.seek(first_frame_offset)
                
                extracted_count = 0
                while True:
                    try:
                        frame_data = self._extract_frame(f)
                        if frame_data is None:
                            break
                        
                        frame_id, jpeg_data = frame_data
                        
                        # Validate JPEG data
                        if not self._is_valid_jpeg(jpeg_data):
                            if self.verbose:
                                print(f"  Warning: Invalid JPEG data for {frame_id}, skipping...")
                            continue
                        
                        output_file = output_dir / f"{frame_id}.jpg"
                        with open(output_file, 'wb') as out_f:
                            out_f.write(jpeg_data)
                        
                        extracted_count += 1
                        if self.verbose and extracted_count % 100 == 0:
                            print(f"  Extracted {extracted_count} frames...")
                            
                    except (struct.error, UnicodeDecodeError) as e:
                        if self.verbose:
                            print(f"  Reached end of valid data at position {f.tell()}")
                        break
                    except Exception as e:
                        if self.verbose:
                            print(f"  Error extracting frame: {e}")
                        continue
                
                if self.verbose:
                    print(f"  Successfully extracted {extracted_count} frames")
                    if frame_count > 0 and extracted_count != frame_count:
                        print(f"  Warning: Expected {frame_count} frames, extracted {extracted_count}")
                
                return extracted_count > 0
                
            except Exception as e:
                if self.verbose:
                    print(f"  Error extracting frames: {e}")
                return False
    
    def _read_header(self, file) -> Tuple[int, int]:
        """Read and validate the MLCF container header.
        
        Args:
            file: Open file object positioned at start
            
        Returns:
            Tuple of (frame_count, first_frame_offset)
            
        Raises:
            ValueError: If magic bytes are invalid
        """
        magic = file.read(4)
        if magic != b'MLCF':
            raise ValueError(f"Invalid magic bytes: expected b'MLCF', got {magic}")
        
        frame_count = struct.unpack('<I', file.read(4))[0]
        first_frame_offset = struct.unpack('<I', file.read(4))[0]
        
        return frame_count, first_frame_offset
    
    def _extract_frame(self, file) -> Optional[Tuple[str, bytes]]:
        """Extract a single frame from the current file position.
        
        Args:
            file: Open file object
            
        Returns:
            Tuple of (frame_id, jpeg_data) or None if end of file
        """
        frame_size_data = file.read(4)
        if len(frame_size_data) < 4:
            return None
        
        frame_size = struct.unpack('<I', frame_size_data)[0]
        frame_id_length = struct.unpack('<B', file.read(1))[0]
        frame_id = file.read(frame_id_length).decode('utf-8')
        
        # Calculate JPEG data size
        jpeg_size = frame_size - 1 - frame_id_length
        jpeg_data = file.read(jpeg_size)
        
        return frame_id, jpeg_data
    
    def _is_valid_jpeg(self, data: bytes) -> bool:
        """Check if data appears to be valid JPEG.
        
        Args:
            data: Bytes to validate
            
        Returns:
            True if data starts with JPEG magic bytes
        """
        return len(data) >= 4 and data[:2] == b'\xff\xd8'