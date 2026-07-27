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

  # on-demand workers
  instance_type_count = {
    # "g4dn.xlarge" = 1,
    # "g6.12xlarge" = 2,
  }

  # persistent spot workers: 용량이 없어도 요청이 열린 채 유지되고,
  # 용량이 생기는 즉시 자동으로 기동된다 (terraform destroy 로 요청 취소).
  spot_instance_type_count = {
    "g6.xlarge" = 1, 
  }
  spot_price = var.spot_price
}
