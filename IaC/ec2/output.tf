output "instance_ids" {
  description = "Map of instance IDs by instance type and index"
  value = {
    for key, instance in aws_instance.worker-instance :
    key => instance.id
  }
}

output "instance_public_ips" {
  description = "Map of public IPs by instance type and index"
  value = {
    for key, instance in aws_instance.worker-instance :
    key => instance.public_ip
  }
}

output "instance_private_ips" {
  description = "Map of private IPs by instance type and index"
  value = {
    for key, instance in aws_instance.worker-instance :
    key => instance.private_ip
  }
}

output "head_instance_public_ip" {
  description = "Public IP of the head instance (null if not created)"
  value = one(aws_instance.head-instance[*].public_ip)
}

output "head_instance_private_ip" {
  description = "Private IP of the head instance (null if not created)"
  value = one(aws_instance.head-instance[*].private_ip)
}

output "spot_request_ids" {
  description = "Map of spot request IDs (sir-...) by instance type and index"
  value = {
    for key, req in aws_spot_instance_request.spot-worker :
    key => req.id
  }
}

output "spot_instance_ids" {
  description = "Map of spot instance IDs by instance type and index"
  value = {
    for key, req in aws_spot_instance_request.spot-worker :
    key => req.spot_instance_id
  }
}

output "spot_instance_public_ips" {
  description = "Map of spot public IPs by instance type and index"
  value = {
    for key, req in aws_spot_instance_request.spot-worker :
    key => req.public_ip
  }
}

output "spot_instance_private_ips" {
  description = "Map of spot private IPs by instance type and index"
  value = {
    for key, req in aws_spot_instance_request.spot-worker :
    key => req.private_ip
  }
}
