output "head_instance_public_ip" {
  description = "Public IP of the head instance"
  value       = module.ec2.head_instance_public_ip
}

output "head_instance_private_ip" {
  description = "Private IP of the head instance"
  value       = module.ec2.head_instance_private_ip
}

output "instance_ids" {
  description = "Map of instance IDs by instance type and index"
  value       = module.ec2.instance_ids
}

output "instance_public_ips" {
  description = "Map of public IPs by instance type and index"
  value       = module.ec2.instance_public_ips
}

output "instance_private_ips" {
  description = "Map of private IPs by instance type and index"
  value       = module.ec2.instance_private_ips
}

output "spot_request_ids" {
  description = "Map of spot request IDs by instance type and index"
  value       = module.ec2.spot_request_ids
}

output "spot_instance_ids" {
  description = "Map of spot instance IDs by instance type and index"
  value       = module.ec2.spot_instance_ids
}

output "spot_instance_public_ips" {
  description = "Map of spot public IPs by instance type and index"
  value       = module.ec2.spot_instance_public_ips
}

output "spot_instance_private_ips" {
  description = "Map of spot private IPs by instance type and index"
  value       = module.ec2.spot_instance_private_ips
}
