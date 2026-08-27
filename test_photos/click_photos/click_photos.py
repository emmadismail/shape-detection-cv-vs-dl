#!/usr/bin/env python3
"""
Raspberry Pi Camera Photo Clicker
Takes photos from multiple Raspberry Pi cameras in parallel and saves them locally.
"""

import configparser
import concurrent.futures
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

class PhotoClicker:
    def __init__(self, hosts_file: Path, base_output_dir: Path, password: str = None):
        self.password = password or os.environ.get("PI_PASSWORD", "")
        self.hosts_file = hosts_file
        self.base_output_dir = base_output_dir
        self.pi_ips: List[str] = []
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_output_dir / self.timestamp
        self.results = {}

    def setup(self):
        """Create session directory and load IPs."""
        # Load IPs
        if not self.hosts_file.exists():
            print(f"❌ hosts.ini not found at {self.hosts_file}")
            sys.exit(1)
            
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(self.hosts_file)
        
        if "raspberry_pis" in config:
            self.pi_ips = [key for key in config["raspberry_pis"].keys() if "." in key]
        
        if not self.pi_ips:
            print(f"❌ No IPs found in hosts.ini at {self.hosts_file}")
            sys.exit(1)
            
        print(f"📋 Found {len(self.pi_ips)} cameras to capture from.")

        # Create directory
        self.session_dir.mkdir(parents=True, exist_ok=True)
        print(f"📂 Saving photos to: {self.session_dir}")

    def take_photo(self, ip: str) -> Tuple[bool, str]:
        """Take photo on a single Pi and stream it back directly (no temp files)."""
        print(f"📸 Starting capture on {ip}...")
        
        # Local image path
        local_filename = f"cam_{ip.replace('.', '_')}.jpg"
        local_path = self.session_dir / local_filename

        # SSH options - optimized for speed
        ssh_opts = [
            "-o", "ConnectTimeout=5",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-o", "BatchMode=yes"  # Fail fast if keys don't work (for first attempt)
        ]

        # Command to run on Pi: capture to stdout
        # -t 2000: 2s delay for AWB/AE
        # -o -: output to stdout
        # --nopreview: no preview window
        remote_cmd = "rpicam-jpeg -o - -t 2000 --nopreview"
        
        # Helper to run SSH command and get binary output
        def run_ssh_capture(use_password=False):
            cmd = []
            final_opts = list(ssh_opts)
            
            if use_password:
                cmd = ["sshpass", "-p", self.password]
                # Remove BatchMode=yes and its preceding "-o"
                # ssh_opts is a flat list: ["-o", "Option", "-o", "Option"...]
                # We need to carefuly remove both "-o" and "BatchMode=yes"
                if "BatchMode=yes" in final_opts:
                    idx = final_opts.index("BatchMode=yes")
                    if idx > 0 and final_opts[idx-1] == "-o":
                        del final_opts[idx]   # Remove BatchMode=yes
                        del final_opts[idx-1] # Remove -o
            
            cmd += ["ssh"] + final_opts + [f"pi@{ip}", remote_cmd]
            
            return subprocess.run(cmd, capture_output=True)

        # Retry wrapper for the whole operation
        max_retries = 3
        for attempt in range(max_retries):
            # Attempt 1: Try with SSH Keys (fastest)
            res = run_ssh_capture(use_password=False)
            if res is None and self.password:
                res = run_ssh_capture(use_password=True)
            
            # Attempt 2: If failed, try with Password (fallback)
            if res.returncode != 0:
                res = run_ssh_capture(use_password=True)

            if res.returncode == 0:
                 # Success! Save stdout to file
                image_data = res.stdout
                if not image_data:
                     # Empty image data, treat as failure but maybe retriable?
                     # Ideally we should retry, but for now let's just log and retry
                     print(f"⚠️ {ip}: Empty image data on attempt {attempt+1}. Retrying...")
                else:
                    with open(local_path, "wb") as f:
                        f.write(image_data)
                    return True, str(local_path)
            
            # If we are here, capture failed
            error_msg = res.stderr.decode('utf-8', errors='ignore').strip()
            
            # Identify if it's a transient SSH/Permission error we should retry
            # "Permission denied", "Connection timed out", "Connection closed", "kex_exchange_identification"
            is_retriable = any(x in error_msg for x in ["Permission denied", "Connection", "kex_exchange", "No route", "reset"])
            
            if attempt < max_retries - 1 and is_retriable:
                # Random jitter backoff to desynchronize
                import random
                sleep_time = random.uniform(0.5, 2.0)
                print(f"⚠️ {ip}: Capture failed (Attempt {attempt+1}/{max_retries}). Retrying in {sleep_time:.1f}s... Error: {error_msg[:50]}")
                time.sleep(sleep_time)
            else:
                 # Last attempt failed
                 return False, f"Capture failed after {max_retries} attempts. Last error: {error_msg}"
        
        return False, "Capture failed (MAX RETRIES)"

    def run(self):
        self.setup()
        
        # Concurrency: we can now safely use more threads as we only do 1 SSH connection per camera
        # and we don't need SCP which has higher overhead.
        max_workers = len(self.pi_ips)
        print(f"🚀 Launching stream capture on {len(self.pi_ips)} cameras (concurrency: {max_workers})...")
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ip = {executor.submit(self.take_photo, ip): ip for ip in self.pi_ips}
            
            for future in concurrent.futures.as_completed(future_to_ip):
                ip = future_to_ip[future]
                try:
                    success, msg = future.result()
                    if success:
                        print(f"✅ {ip}: Saved photo")
                        self.results[ip] = "Success"
                    else:
                        print(f"❌ {ip}: {msg}")
                        self.results[ip] = f"Failed: {msg}"
                except Exception as exc:
                    print(f"❌ {ip}: Exception {exc}")
                    self.results[ip] = f"Error: {exc}"

        duration = time.time() - start_time
        
        # Summary
        success_count = sum(1 for v in self.results.values() if v == "Success")
        print("\n" + "="*50)
        print(f"🏁 Completed in {duration:.1f} seconds")
        print(f"✅ Successful: {success_count}/{len(self.pi_ips)}")
        print(f"❌ Failed: {len(self.pi_ips) - success_count}")
        print(f"📂 Photos saved in: {self.session_dir}")
        print("="*50)

if __name__ == "__main__":
    # Determine paths
    # Assume this script is in pi_on_wifi/click_photos/
    # hosts.ini is in pi_on_wifi/
    
    script_file = Path(__file__).resolve()
    script_dir = script_file.parent
    
    # Check if hosts.ini is in parent dir
    #hosts_path = script_dir.parent / "hosts.ini"
    hosts_path = script_dir / "hosts.ini"
    
    
    if not hosts_path.exists():
        # Fallback check current dir (if run from wrong location)
        if Path("hosts.ini").exists():
            hosts_path = Path("hosts.ini").resolve()
            
    clicker = PhotoClicker(hosts_path, script_dir)
    clicker.run()
