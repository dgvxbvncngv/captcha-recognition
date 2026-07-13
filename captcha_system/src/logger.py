# src/logger.py
# 操作日志记录模块
from src.db import get_connection


def log_action(user_id, username, action, detail=None, ip=None):
    """记录一条操作日志"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                '''INSERT INTO operation_logs (user_id, username, action, detail, ip)
                   VALUES (%s,%s,%s,%s,%s)''',
                (user_id, username, action, detail, ip),
            )
        conn.commit()
    except Exception:
        # 日志失败不影响主流程
        pass
    finally:
        conn.close()


def query_logs(user_id=None, action=None, start_date=None, end_date=None, limit=100, offset=0):
    """查询操作日志"""
    sql = 'SELECT id, user_id, username, action, detail, ip, created_at FROM operation_logs WHERE 1=1'
    params = []
    if user_id is not None:
        sql += ' AND user_id=%s'
        params.append(user_id)
    if action:
        sql += ' AND action=%s'
        params.append(action)
    if start_date:
        sql += ' AND created_at>=%s'
        params.append(start_date)
    if end_date:
        sql += ' AND created_at<=%s'
        params.append(end_date)
    sql += ' ORDER BY created_at DESC LIMIT %s OFFSET %s'
    params.extend([limit, offset])

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()
