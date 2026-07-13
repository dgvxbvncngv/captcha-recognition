# src/auth.py
# 用户认证模块：注册、登录、登出、密码哈希
import hashlib
import os
from src.db import get_connection


def _hash_password(password: str, salt: str = None) -> str:
    """PBKDF2-HMAC-SHA256 哈希密码，返回 'salt$hash' 字符串"""
    if salt is None:
        salt = os.urandom(16).hex()
    dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), bytes.fromhex(salt), 100000)
    return f'{salt}${dk.hex()}'


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt, _ = stored.split('$', 1)
        return _hash_password(password, salt) == stored
    except Exception:
        return False


def register(username: str, password: str, role: str = 'user') -> dict:
    """注册新用户。返回 {'ok': bool, 'msg': str, 'user': dict}"""
    username = (username or '').strip()
    if not username or not password:
        return {'ok': False, 'msg': '用户名和密码不能为空'}
    if len(username) < 3 or len(username) > 50:
        return {'ok': False, 'msg': '用户名长度需 3~50 字符'}
    if len(password) < 6:
        return {'ok': False, 'msg': '密码至少 6 位'}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT id FROM users WHERE username=%s', (username,))
            if cur.fetchone():
                return {'ok': False, 'msg': '用户名已存在'}
            # 第一个注册的用户设为管理员
            cur.execute('SELECT COUNT(*) AS c FROM users')
            if cur.fetchone()['c'] == 0:
                role = 'admin'
            cur.execute(
                'INSERT INTO users (username, password_hash, role) VALUES (%s,%s,%s)',
                (username, _hash_password(password), role),
            )
            cur.execute('SELECT id, username, role, created_at FROM users WHERE username=%s', (username,))
            user = cur.fetchone()
        conn.commit()
        return {'ok': True, 'msg': '注册成功', 'user': user}
    finally:
        conn.close()


def login(username: str, password: str) -> dict:
    """登录验证。返回 {'ok': bool, 'msg': str, 'user': dict}"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT id, username, password_hash, role, created_at FROM users WHERE username=%s',
                (username,),
            )
            row = cur.fetchone()
            if not row or not _verify_password(password, row['password_hash']):
                return {'ok': False, 'msg': '用户名或密码错误'}
            user = {'id': row['id'], 'username': row['username'], 'role': row['role'], 'created_at': row['created_at']}
        return {'ok': True, 'msg': '登录成功', 'user': user}
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> dict:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT id, username, role, created_at FROM users WHERE id=%s', (user_id,))
            return cur.fetchone()
    finally:
        conn.close()


def list_users() -> list:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT id, username, role, created_at FROM users ORDER BY id')
            return cur.fetchall()
    finally:
        conn.close()
