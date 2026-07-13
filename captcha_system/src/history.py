# src/history.py
# 识别历史记录存储与查询
from src.db import get_connection


def save_history(user_id, image_name, image_data, predicted_text, confidence,
                 is_correct=None, captcha_type='alphanumeric'):
    """保存一条识别记录"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                '''INSERT INTO recognition_history
                   (user_id, image_name, image_data, predicted_text, confidence,
                    is_correct, captcha_type)
                   VALUES (%s,%s,%s,%s,%s,%s,%s)''',
                (user_id, image_name, image_data, predicted_text, confidence,
                 is_correct, captcha_type),
            )
            new_id = cur.lastrowid
        conn.commit()
        return new_id
    finally:
        conn.close()


def query_history(user_id=None, captcha_type=None, start_date=None, end_date=None,
                  limit=50, offset=0, exclude_image=True):
    """按条件查询历史记录。exclude_image=True 时不返回图片二进制"""
    fields = 'id, user_id, image_name, predicted_text, confidence, is_correct, captcha_type, created_at'
    if not exclude_image:
        fields += ', image_data'
    sql = f'SELECT {fields} FROM recognition_history WHERE 1=1'
    params = []
    if user_id is not None:
        sql += ' AND user_id=%s'
        params.append(user_id)
    if captcha_type:
        sql += ' AND captcha_type=%s'
        params.append(captcha_type)
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
            rows = cur.fetchall()
        return rows
    finally:
        conn.close()


def get_history_image(history_id):
    """获取历史记录的图片二进制"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT image_data, image_name, user_id FROM recognition_history WHERE id=%s', (history_id,))
            return cur.fetchone()
    finally:
        conn.close()


def get_statistics(user_id=None, days=30):
    """获取统计数据用于图表"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            base_where = 'WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)'
            params = [days]
            if user_id is not None:
                base_where += ' AND user_id=%s'
                params.append(user_id)

            # 每日识别数量与平均置信度
            cur.execute(
                f'''SELECT DATE(created_at) AS day,
                       COUNT(*) AS count,
                       AVG(confidence) AS avg_conf,
                       SUM(CASE WHEN is_correct=1 THEN 1 ELSE 0 END) AS correct_count
                   FROM recognition_history {base_where}
                   GROUP BY DATE(created_at)
                   ORDER BY day''',
                params,
            )
            daily = cur.fetchall()

            # 验证码类型分布
            cur.execute(
                f'''SELECT captcha_type, COUNT(*) AS count
                   FROM recognition_history {base_where}
                   GROUP BY captcha_type''',
                params,
            )
            type_dist = cur.fetchall()

            # 总体统计
            cur.execute(
                f'''SELECT COUNT(*) AS total,
                       AVG(confidence) AS avg_conf,
                       SUM(CASE WHEN is_correct=1 THEN 1 ELSE 0 END) AS correct
                   FROM recognition_history {base_where}''',
                params,
            )
            overall = cur.fetchone()
        return {'daily': daily, 'type_distribution': type_dist, 'overall': overall}
    finally:
        conn.close()


def get_history_by_id(history_id):
    """根据 ID 获取单条历史记录"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT id, user_id, image_name, predicted_text, confidence, '
                'is_correct, captcha_type, created_at FROM recognition_history WHERE id=%s',
                (history_id,),
            )
            return cur.fetchone()
    finally:
        conn.close()
