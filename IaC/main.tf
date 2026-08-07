module "vpc" {
  source = "./vpc"

  prefix              = var.prefix
  public_subnet_cidrs = var.public_subnet_cidrs
  azs                 = [for az in var.azs : "${var.region}${az}"]
}

module "ec2" {
  source = "./ec2"

  ami_id                   = var.ami_id
  key_name                 = var.key_name
  prefix                   = var.prefix
  head_instance_type       = var.head_instance_type
  security_group_id        = module.vpc.security_group_id
  subnet_id                = module.vpc.public_subnet_ids[0]
  s3_instance_profile_name = aws_iam_instance_profile.s3-instance-profile.name
  root_volume_size         = var.root_volume_size

  instance_type_count      = var.instance_type_count
  spot_instance_type_count = var.spot_instance_type_count
  spot_price               = var.spot_price
}
