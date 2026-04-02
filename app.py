"""
医疗数据离群值检测系统 - MVC架构主应用
"""
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import os

# 导入控制器
from controllers import AnalysisController

# 创建Flask应用
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# 创建控制器实例
analysis_controller = AnalysisController()


@app.route('/')
def index():
    """主页 - 返回前端页面"""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """分析API端点"""
    try:
        data = request.json
        analysis_type = data.get('analysisType', 'full')
        model_select = data.get('modelSelect', 'all')
        
        # 调用控制器处理业务逻辑
        result = analysis_controller.analyze(analysis_type, model_select)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({'status': 'ok', 'message': '系统运行正常'})


if __name__ == '__main__':
    # 创建必要的目录
    os.makedirs('results', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    
    print("="*60)
    print("医疗数据离群值检测系统 - MVC架构")
    print("="*60)
    print("目录结构:")
    print("  - models/      : 数据模型层")
    print("  - views/       : 视图层")
    print("  - controllers/ : 控制器层")
    print("  - static/      : 静态资源")
    print("  - templates/   : HTML模板")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)