# file: web_server.py
# directory: .
from flask import Flask, request, jsonify
from utils import configure_thread_logging

def start_web_server(page_queue, stats_collector, port, log_level, log_output):
    app = Flask(__name__)

    # Set up logger
    log_filename = 'logs/web_server/web_server.log'
    logger = configure_thread_logging('web_server', log_filename, log_level, log_output)

    @app.route('/process_page', methods=['POST'])
    def process_page():
        h = request.args.get('h', type=int)
        if h is None:
            return jsonify({'error': 'Parameter h is required'}), 400
        page_queue.put(h)
        return jsonify({'status': f'Page {h} queued for processing'}), 200

    @app.route('/process_pages', methods=['POST'])
    def process_pages():
        h = request.args.get('h', type=int)
        k = request.args.get('k', type=int)
        m = request.args.get('m', type=int)
        n = request.args.get('n', type=int)
        if h is None or k is None:
            return jsonify({'error': 'Parameters h and k are required'}), 400
        if m is not None and n is not None:
            for page_num in range(h, k+1):
                if (page_num % m) == n:
                    page_queue.put(page_num)
        else:
            for page_num in range(h, k+1):
                page_queue.put(page_num)
        return jsonify({'status': f'Pages {h} to {k} queued for processing'}), 200

    @app.route('/stats', methods=['GET'])
    def get_stats():
        stats = stats_collector.reset()
        return jsonify(stats), 200

    # Run the Flask app
    app.run(host='0.0.0.0', port=port, threaded=True)
