# src/db.py
# MySQL 数据库连接与初始化
import pymysql
from src.db_config import MYSQL_CONFIG


def get_connection(database=True):
    """获取数据库连接"""
    cfg = MYSQL_CONFIG.copy()
    if not database:
        cfg.pop('database')
    return pymysql.connect(
        host=cfg['host'],
        port=cfg['port'],
        user=cfg['user'],
        password=cfg['password'],
        charset=cfg['charset'],
        database=cfg.get('database'),
        cursorclass=pymysql.cursors.DictCursor,
    )


def init_db():
    """创建数据库和所有表（如不存在）"""
    # 1. 创建数据库
    conn = get_connection(database=False)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE DATABASE IF NOT EXISTS captcha_system "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        conn.commit()
    finally:
        conn.close()

    # 2. 创建表
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # 用户表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) NOT NULL UNIQUE,
                    password_hash VARCHAR(255) NOT NULL,
                    role ENUM('admin', 'user') NOT NULL DEFAULT 'user',
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)

            # 识别历史记录表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS recognition_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NULL,
                    image_name VARCHAR(255),
                    image_data LONGBLOB,
                    predicted_text VARCHAR(64),
                    confidence FLOAT,
                    is_correct TINYINT(1) DEFAULT 0 COMMENT '1=正确,0=错误,NULL=未知',
                    captcha_type VARCHAR(32) DEFAULT 'alphanumeric' COMMENT '验证码类型',
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_user (user_id),
                    INDEX idx_created (created_at),
                    INDEX idx_type (captcha_type)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)

            # 操作日志表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS operation_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NULL,
                    username VARCHAR(50),
                    action VARCHAR(64) NOT NULL COMMENT '操作类型',
                    detail TEXT,
                    ip VARCHAR(64),
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_user (user_id),
                    INDEX idx_action (action),
                    INDEX idx_created (created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)

        conn.commit()
    finally:
        conn.close()


if __name__ == '__main__':
    init_db()
    print('数据库初始化完成')
