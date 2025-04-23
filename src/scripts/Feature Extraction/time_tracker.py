from config import *


class TimeTracker:
    """Class to track processing time and provide estimates"""
    
    def __init__(self, total_files):
        self.start_time = time.time()
        self.total_files = total_files
        self.completed = 0
        self.skipped = 0
        self.failed = 0
        self.file_times = []  # Store processing times for better estimates
        self.last_print_time = time.time()
        
    def update(self, status, processing_time=None):
        """Update tracker with a new file status"""
        if status == 'completed':
            self.completed += 1
            if processing_time:
                self.file_times.append(processing_time)
        elif status == 'skipped':
            self.skipped += 1
        elif status == 'failed':
            self.failed += 1
        
    def get_progress_str(self):
        """Get a formatted progress string with ETA"""
        elapsed = time.time() - self.start_time
        total_processed = self.completed + self.skipped + self.failed
        progress = total_processed / self.total_files if self.total_files > 0 else 0
        
        # Calculate ETA based on average of recent processing times
        if len(self.file_times) > 0 and self.completed > 0:
            # Use up to the last 10 files for a more accurate recent average
            recent_times = self.file_times[-min(10, len(self.file_times)):]
            avg_time_per_file = sum(recent_times) / len(recent_times)
            files_remaining = self.total_files - total_processed
            eta_seconds = avg_time_per_file * files_remaining
        else:
            # Fall back to simple estimation if we don't have timing data
            if progress > 0 and elapsed > 0:
                eta_seconds = (elapsed / progress) - elapsed
            else:
                eta_seconds = 0
        
        # Format ETA string
        if eta_seconds > 0:
            eta_td = timedelta(seconds=int(eta_seconds))
            if eta_td.days > 0:
                eta_str = f"{eta_td.days}d {eta_td.seconds//3600}h {(eta_td.seconds//60)%60}m"
            elif eta_td.seconds > 3600:
                eta_str = f"{eta_td.seconds//3600}h {(eta_td.seconds//60)%60}m {eta_td.seconds%60}s"
            elif eta_td.seconds > 60:
                eta_str = f"{eta_td.seconds//60}m {eta_td.seconds%60}s"
            else:
                eta_str = f"{eta_td.seconds}s"
        else:
            eta_str = "calculating..."
            
        # Format elapsed time string
        elapsed_td = timedelta(seconds=int(elapsed))
        if elapsed_td.days > 0:
            elapsed_str = f"{elapsed_td.days}d {elapsed_td.seconds//3600}h {(elapsed_td.seconds//60)%60}m"
        elif elapsed_td.seconds > 3600:
            elapsed_str = f"{elapsed_td.seconds//3600}h {(elapsed_td.seconds//60)%60}m {elapsed_td.seconds%60}s"
        elif elapsed_td.seconds > 60:
            elapsed_str = f"{elapsed_td.seconds//60}m {elapsed_td.seconds%60}s"
        else:
            elapsed_str = f"{elapsed_td.seconds}s"
        
        # Build the progress string
        progress_str = (f"[{total_processed}/{self.total_files}] {progress:.1%} "
                        f"(✅{self.completed} ⏩{self.skipped} ❌{self.failed}) "
                        f"Elapsed: {elapsed_str} | ETA: {eta_str}")
                        
        # If we have enough data, add throughput
        if elapsed > 60 and self.completed > 0:  # Minimum 1 minute elapsed
            files_per_minute = (self.completed / elapsed) * 60
            progress_str += f" | Rate: {files_per_minute:.1f} files/min"
            
        return progress_str
    
    def should_print_update(self):
        """Check if we should print a progress update"""
        current_time = time.time()
        if current_time - self.last_print_time >= PRINT_INTERVAL:
            self.last_print_time = current_time
            return True
        return False
        
    def get_final_stats(self):
        """Get final statistics string"""
        elapsed = time.time() - self.start_time
        elapsed_td = timedelta(seconds=int(elapsed))
        
        # Format elapsed time
        if elapsed_td.days > 0:
            elapsed_str = f"{elapsed_td.days}d {elapsed_td.seconds//3600}h {(elapsed_td.seconds//60)%60}m"
        elif elapsed_td.seconds > 3600:
            elapsed_str = f"{elapsed_td.seconds//3600}h {(elapsed_td.seconds//60)%60}m {elapsed_td.seconds%60}s"
        elif elapsed_td.seconds > 60:
            elapsed_str = f"{elapsed_td.seconds//60}m {elapsed_td.seconds%60}s"
        else:
            elapsed_str = f"{elapsed_td.seconds}s"
            
        # Calculate average time per file
        if self.completed > 0:
            avg_time = sum(self.file_times) / len(self.file_times)
            avg_time_str = f"{avg_time:.2f}s"
        else:
            avg_time_str = "N/A"
            
        # Build the stats string
        stats = [
            f"Total time: {elapsed_str}",
            f"Completed:  {self.completed} files",
            f"Skipped:    {self.skipped} files",
            f"Failed:     {self.failed} files",
            f"Total:      {self.total_files} files",
            f"Avg. time:  {avg_time_str} per file"
        ]
        
        if elapsed > 60 and self.completed > 0:
            files_per_minute = (self.completed / elapsed) * 60
            stats.append(f"Throughput:  {files_per_minute:.1f} files/min")
            
        return "\n".join(stats)
