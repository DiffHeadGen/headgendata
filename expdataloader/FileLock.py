import fcntl
import os


class FileLock:
    def __init__(self, lock_file):
        self.lock_file = lock_file
        self.lock_fd = None

        # 如果锁文件不存在，创建空文件
        if not os.path.exists(lock_file):
            open(lock_file, "w").close()

    def acquire(self):
        """尝试获取文件锁"""
        try:
            self.lock_fd = open(self.lock_file, "w")
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except IOError:
            return False

    def release(self):
        """释放文件锁"""
        if self.lock_fd:
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            self.lock_fd.close()
            self.lock_fd = None
            os.remove(self.lock_file)  # 删除文件锁