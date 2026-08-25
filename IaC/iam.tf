resource "aws_iam_role" "s3-role" {
    name = "${var.prefix}-s3-role"
    assume_role_policy = jsonencode({
        Version = "2012-10-17"
        Statement = [
            {
                Action = "sts:AssumeRole"
                Effect = "Allow"
                Principal = {
                    Service = "ec2.amazonaws.com"
                }
            }
        ]
    })
}

# var.s3_bucket_names 에 적힌 버킷들만 접근 가능 (빈 리스트면 S3 권한 없음)
resource "aws_iam_role_policy" "s3-access" {
    count = length(var.s3_bucket_names) > 0 ? 1 : 0

    name = "${var.prefix}-s3-access"
    role = aws_iam_role.s3-role.id
    policy = jsonencode({
        Version = "2012-10-17"
        Statement = [
            {
                Action = "s3:*"
                Effect = "Allow"
                Resource = flatten([
                    for bucket in var.s3_bucket_names : [
                        "arn:aws:s3:::${bucket}",
                        "arn:aws:s3:::${bucket}/*",
                    ]
                ])
            }
        ]
    })
}

resource "aws_iam_instance_profile" "s3-instance-profile" {
    name = "${var.prefix}-s3-instance-profile"
    role = aws_iam_role.s3-role.name
}
