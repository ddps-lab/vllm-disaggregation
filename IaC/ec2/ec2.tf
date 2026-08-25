# 키는 "<type>-<az>-<같은 (type, az) 안에서의 순번>" (e.g. "g6.xlarge-b-0").
# 인덱스가 전체 리스트 위치가 아니라 AZ별 순번이므로, 리스트에서 다른 AZ 항목을
# 지우거나 순서를 바꿔도 살아남는 인스턴스의 키가 유지된다 (재생성 방지).
locals {
  instances = merge([
    for instance_type, azs in var.instance_type_azs : {
      for i, az in azs :
      "${instance_type}-${az != "" ? az : "default"}-${length([for j in range(i) : j if azs[j] == az])}" => { type = instance_type, az = az }
    }
  ]...)

  spot_instances = merge([
    for instance_type, azs in var.spot_instance_type_azs : {
      for i, az in azs :
      "${instance_type}-${az != "" ? az : "default"}-${length([for j in range(i) : j if azs[j] == az])}" => { type = instance_type, az = az }
    }
  ]...)
}

resource "aws_spot_instance_request" "spot-worker" {
  for_each = local.spot_instances

  ami                    = var.ami_id
  instance_type          = each.value.type
  key_name               = var.key_name
  monitoring             = true
  subnet_id              = each.value.az != "" ? var.subnet_id_by_az[each.value.az] : var.subnet_id
  vpc_security_group_ids = [var.security_group_id]

  iam_instance_profile = var.s3_instance_profile_name

  # persistent: 요청이 취소될 때까지 열려 있어 용량이 생기면 자동 기동되고,
  # 중단(interruption) 시 stop 됐다가 용량 복귀 시 자동 재시작된다.
  spot_type                      = "persistent"
  instance_interruption_behavior = "stop"

  # spot 인스턴스가 실제로 기동될 때까지 apply 가 대기한다 (최대 아래 timeouts).
  # 기동 성공 후에야 head/on-demand 워커 생성이 시작된다 (depends_on).
  wait_for_fulfillment = true

  timeouts {
    create = "1h"
  }

  spot_price = var.spot_price != "" ? var.spot_price : null

  root_block_device {
    volume_size = var.root_volume_size
  }

  # 주의: aws_spot_instance_request 의 tags 는 spot "요청"에 붙는다.
  # 기동된 인스턴스에는 자동 전파되지 않는다 (아래 aws_ec2_tag 로 부착).
  tags = {
    Name         = "${var.prefix}-spot-worker-${each.key}"
    InstanceType = each.value.type
    Index        = split("-", each.key)[length(split("-", each.key)) - 1]
  }
}

# spot 요청의 tags 가 인스턴스로 전파되지 않으므로,
# 기동된 인스턴스에 Name 태그를 직접 부착한다.
# wait_for_fulfillment = true 라 apply 시점에 spot_instance_id 를 알 수 있다.
# 인스턴스가 교체되면(수동 terminate 후 재이행 등) apply 를 다시 실행해야 재부착된다.
resource "aws_ec2_tag" "spot-worker-name" {
  for_each = aws_spot_instance_request.spot-worker

  resource_id = each.value.spot_instance_id
  key         = "Name"
  value       = "${var.prefix}-spot-worker-${each.key}"
}

# 고정 IP: 인스턴스가 교체/재시작돼도 EIP 주소는 유지된다.
# 교체 후 apply 를 실행하면 같은 EIP 가 새 인스턴스에 재연결된다.
resource "aws_eip" "spot-worker" {
  for_each = local.spot_instances

  domain = "vpc"

  tags = {
    Name = "${var.prefix}-spot-worker-${each.key}"
  }
}

resource "aws_eip_association" "spot-worker" {
  for_each = aws_spot_instance_request.spot-worker

  instance_id   = each.value.spot_instance_id
  allocation_id = aws_eip.spot-worker[each.key].id
}

resource "aws_instance" "head-instance" {
  # head_instance_type 이 빈 문자열이면 head 를 만들지 않는다
  count = var.head_instance_type != "" ? 1 : 0

  # spot 워커가 성공적으로 기동된 뒤에만 생성
  depends_on = [aws_spot_instance_request.spot-worker]

  ami                    = var.ami_id
  instance_type          = var.head_instance_type
  key_name               = var.key_name
  monitoring             = true
  subnet_id              = var.subnet_id
  vpc_security_group_ids = [var.security_group_id]

  iam_instance_profile = var.s3_instance_profile_name

  root_block_device {
    volume_size = var.root_volume_size
  }

  tags = {
    Name = "${var.prefix}-head-instance"
  }
}

resource "aws_instance" "worker-instance" {
  # spot 워커가 성공적으로 기동된 뒤에만 생성
  depends_on = [aws_spot_instance_request.spot-worker]

  for_each = local.instances

  ami                    = var.ami_id
  instance_type          = each.value.type
  key_name               = var.key_name
  monitoring             = true
  subnet_id              = each.value.az != "" ? var.subnet_id_by_az[each.value.az] : var.subnet_id
  vpc_security_group_ids = [var.security_group_id]

  iam_instance_profile = var.s3_instance_profile_name

  root_block_device {
    volume_size = var.root_volume_size
  }

  tags = {
    Name         = "${var.prefix}-worker-${each.key}"
    InstanceType = each.value.type
    Index        = split("-", each.key)[length(split("-", each.key)) - 1]
  }
}
