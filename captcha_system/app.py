# app.py
# 网站验证码识别系统 - Flask 主应用
import os
import datetime as dt
import decimal
from functools import wraps

from flask import (
    Flask, request, jsonify, render_template,
    redirect, url_for, session, Response,
)
import torch

from src.predict import load_model, predict
from src.db import init_db
from src import auth, history, logger

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'captcha-system-secret-2026')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(device)


# ---------------- 工具函数 ----------------

def current_user():
    """从 session 获取当前登录用户"""
    uid = session.get('uid')
    if not uid:
        return None
    return auth.get_user_by_id(uid)


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not current_user():
            # API 请求返回 401，页面请求重定向
            if request.path.startswith('/api/'):
                return jsonify({'error': '请先登录'}), 401
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapper


def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        user = current_user()
        if not user:
            if request.path.startswith('/api/'):
                return jsonify({'error': '请先登录'}), 401
            return redirect(url_for('login'))
        if user['role'] != 'admin':
            return jsonify({'error': '需要管理员权限'}), 403
        return f(*args, **kwargs)
    return wrapper


def client_ip():
    return request.headers.get('X-Forwarded-For', request.remote_addr or '')


def datetime_to_str(obj):
    """递归把 datetime/Decimal/date 转成可 JSON 序列化的类型"""
    if isinstance(obj, dict):
        return {k: datetime_to_str(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [datetime_to_str(x) for x in obj]
    if isinstance(obj, dt.datetime):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    if isinstance(obj, dt.date):
        return obj.isoformat()
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    return obj


# ---------------- 页面路由 ----------------

@app.route('/')
def index():
    return render_template('index.html', user=current_user())


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        result = auth.login(username, password)
        if result['ok']:
            session['uid'] = result['user']['id']
            logger.log_action(result['user']['id'], username, 'login', '登录成功', client_ip())
            return redirect(url_for('index'))
        return render_template('login.html', error=result['msg'])
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        result = auth.register(username, password)
        if result['ok']:
            logger.log_action(result['user']['id'], username, 'register', '注册成功', client_ip())
            return redirect(url_for('login'))
        return render_template('register.html', error=result['msg'])
    return render_template('register.html')


@app.route('/logout')
def logout():
    user = current_user()
    if user:
        logger.log_action(user['id'], user['username'], 'logout', '登出', client_ip())
    session.pop('uid', None)
    return redirect(url_for('login'))


@app.route('/history')
@login_required
def history_page():
    return render_template('history.html', user=current_user())


@app.route('/stats')
@login_required
def stats_page():
    return render_template('stats.html', user=current_user())


@app.route('/logs')
@admin_required
def logs_page():
    return render_template('logs.html', user=current_user())


# ---------------- API 路由 ----------------

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def api_predict():
    user = current_user()
    if 'image' not in request.files:
        return jsonify({'error': '请上传图片文件'}), 400
    file = request.files['image']
    image_bytes = file.read()
    try:
        text, conf = predict(image_bytes, model, device)
        # 保存历史记录（已登录用户）
        uid = user['id'] if user else None
        try:
            history.save_history(
                user_id=uid,
                image_name=file.filename,
                image_data=image_bytes,
                predicted_text=text,
                confidence=round(conf, 4),
                captcha_type='alphanumeric',
            )
        except Exception as e:
            print('保存历史失败:', e)
        if user:
            logger.log_action(user['id'], user['username'], 'predict',
                              f'结果={text}, 置信度={conf:.4f}', client_ip())
        return jsonify({'result': text, 'confidence': round(conf, 4)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history')
@login_required
def api_history():
    user = current_user()
    # 普通用户只能看自己的，管理员可看全部
    uid = None if user['role'] == 'admin' and request.args.get('all') == '1' else user['id']
    captcha_type = request.args.get('type') or None
    start = request.args.get('start') or None
    end = request.args.get('end') or None
    limit = min(int(request.args.get('limit', 50)), 500)
    offset = max(int(request.args.get('offset', 0)), 0)
    rows = history.query_history(uid, captcha_type, start, end, limit, offset)
    return jsonify(datetime_to_str(rows))


@app.route('/api/history/image/<int:hid>')
@login_required
def api_history_image(hid):
    user = current_user()
    row = history.get_history_image(hid)
    if not row or not row.get('image_data'):
        return jsonify({'error': '图片不存在'}), 404
    # 普通用户只能查看自己的图片
    if user['role'] != 'admin' and row['user_id'] != user['id']:
        return jsonify({'error': '无权访问'}), 403
    return Response(row['image_data'], mimetype='image/png')


@app.route('/api/stats')
@login_required
def api_stats():
    user = current_user()
    uid = None if user['role'] == 'admin' and request.args.get('all') == '1' else user['id']
    days = int(request.args.get('days', 30))
    data = history.get_statistics(uid, days)
    return jsonify(datetime_to_str(data))


@app.route('/api/logs')
@admin_required
def api_logs():
    action = request.args.get('action') or None
    start = request.args.get('start') or None
    end = request.args.get('end') or None
    limit = min(int(request.args.get('limit', 100)), 500)
    offset = max(int(request.args.get('offset', 0)), 0)
    rows = logger.query_logs(None, action, start, end, limit, offset)
    return jsonify(datetime_to_str(rows))


@app.route('/api/users')
@admin_required
def api_users():
    return jsonify(datetime_to_str(auth.list_users()))


if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
