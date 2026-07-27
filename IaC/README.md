# IaC

vLLM disaggregation 실험용 EC2 인프라. 외부 모듈 없이 네이티브 리소스만 사용하며,
on-demand 워커에 더해 **persistent spot 워커**를 지원한다.

## 구조

- `vpc/` — VPC, 퍼블릭 서브넷 4개(AZ a~d), IGW, 라우팅, 보안 그룹
- `ec2/` — spot 워커 + head 인스턴스 + on-demand 워커
- 루트 — provider, IAM(S3 인스턴스 프로파일), 모듈 호출(`main.tf`, `vpc.tf`)

## 구성

- spot 워커: `main.tf` 의 `spot_instance_type_count` 맵 (기본 `g4dn.xlarge` = T4 1대)
- head 인스턴스 1대 (on-demand, 기본 `m5.large`, `head_instance_type = ""` 로 비우면 미생성)
- on-demand 워커: `main.tf` 의 `instance_type_count` 맵

## 사용법

```bash
cp var.tf.sample var.tf   # 값 채우기 (prefix, awscli_profile, ami_id, key_name 등)
terraform init
terraform apply
```

## Spot 동작 방식 (persistent)

- **생성 순서**: spot 워커가 먼저 요청되고, 실제로 기동될 때까지 `apply` 가 대기한다
  (`wait_for_fulfillment = true`, 최대 1시간). **spot 기동이 성공한 뒤에야
  head/on-demand 워커가 생성된다** (`depends_on`).
- 1시간 내 용량을 못 받으면 apply 는 실패하고 on-demand 는 생성되지 않는다.
  재시도는 `terraform apply` 재실행.
- persistent 요청이므로 기동 후 중단(interruption)되면 stop 상태가 됐다가
  용량 복귀 시 자동 재시작된다.
- `terraform destroy` 가 spot 요청 취소 + 인스턴스 종료까지 수행한다.
  **요청만 남겨두면 나중에 인스턴스가 저절로 다시 떠서 과금될 수 있으므로,
  실험 종료 시 반드시 destroy 할 것.**

### 상태 확인

apply 성공 후에는 output 에 IP 가 채워져 있다:

```bash
terraform output spot_instance_public_ips
```

AWS CLI 로 spot 요청 상태를 직접 보려면:

```bash
aws ec2 describe-spot-instance-requests \
  --spot-instance-request-ids $(terraform output -json spot_request_ids | jq -r '.[]') \
  --query 'SpotInstanceRequests[].{state:State,status:Status.Code,instance:InstanceId}'
```

### 주의

- spot 요청의 `tags` 는 요청 객체에만 붙고 기동된 인스턴스에는 전파되지 않는다.
  인스턴스는 위 CLI 의 `InstanceId` 로 찾는다.
- `spot_price` 를 비워두면 on-demand 가격이 상한이 된다 (권장).
- 서브넷 AZ 는 a~d 4개를 쓰므로, c/d AZ 가 없는 리전에서는 `vpc/vpc.tf` 의
  `azs` 를 조정해야 한다 (기본 리전 us-west-2 는 a~d 모두 존재).
