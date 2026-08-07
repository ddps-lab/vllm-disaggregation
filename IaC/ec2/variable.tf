variable "ami_id" {}
variable "key_name" {}
variable "instance_type_azs" {}
variable "prefix" {}
variable "security_group_id" {}
variable "subnet_id" {}
variable "s3_instance_profile_name" {}

variable "head_instance_type" {}
variable "root_volume_size" {}

# AZ suffix("a" 등) => subnet id. 워커의 az 가 "" 가 아니면 이 맵으로 서브넷을 찾는다.
variable "subnet_id_by_az" {
  default = {}
}

variable "spot_instance_type_azs" {
  default = {}
}
variable "spot_price" {
  default = ""
}
