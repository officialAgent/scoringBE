from flask import Blueprint, jsonify, request,send_from_directory
from database.models import DeviceCatalog
from extensions import db

device_bp = Blueprint('device_bp', __name__)

@device_bp.route('/devices', methods=['GET'])
def get_devices():
    devices = DeviceCatalog.query.all()
    return jsonify([{
        'id': d.id,
        'name': d.name,
        'price': float(d.price),
        'status': d.status,
        'description': d.description,
        'image_url': d.image_url,
        'long_description': d.long_description
    } for d in devices])

@device_bp.route('/devices/<int:device_id>', methods=['GET'])
def get_device(device_id):
    device = DeviceCatalog.query.get_or_404(device_id)
    return jsonify({
        'id': device.id,
        'name': device.name,
        'price': float(device.price),
        'status': device.status,
        'description': device.description,
        'image_url': device.image_url,
        'long_description': device.long_description
    })

@device_bp.route('/devices', methods=['POST'])
def add_device():
    data = request.json
    device = DeviceCatalog(
        name=data['name'],
        price=data['price'],
        status=data.get('status', 'Active'),
        description=data.get('description', ''),
        image_url=data.get('image_url', ''),
        long_description=data.get('long_description', '')
    )
    db.session.add(device)
    db.session.commit()
    return jsonify({'message': 'Device added successfully'}), 201

@device_bp.route('/devices/<int:device_id>', methods=['PUT'])
def update_device(device_id):
    data = request.json
    device = DeviceCatalog.query.get_or_404(device_id)
    device.name = data.get('name', device.name)
    device.price = data.get('price', device.price)
    device.status = data.get('status', device.status)
    device.description = data.get('description', device.description)
    device.image_url = data.get('image_url', device.image_url)
    device.long_description = data.get('long_description', device.long_description)
    db.session.commit()
    return jsonify({'message': 'Device updated successfully'})


@device_bp.route('/images/<path:filename>')
def get_image(filename):
    return send_from_directory('assets/img', filename)

