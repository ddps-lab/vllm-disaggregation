variable "ami_id" {}
variable "key_name" {}
variable "instance_type_count" {}
variable "prefix" {}
variable "security_group_id" {}
variable "subnet_id" {}
variable "s3_instance_profile_name" {}

variable "head_instance_type" {}
variable "root_volume_size" {}

variable "spot_instance_type_count" {
  default = {}
}
variable "spot_price" {
  default = ""
}
