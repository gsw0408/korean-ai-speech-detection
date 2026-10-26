# 실험 데모 시작: GPU 인스턴스를 켜고 8080 포트 접속을 허용한 뒤 접속 주소를 출력합니다.
# 기본: 모든 IP에서 접속 가능(발표 PC가 따로 있을 때). 토큰 없는 요청은 서버가 404로 거부합니다.
# -OnlyThisPc: 지금 PC의 공인 IP만 허용합니다.
param([switch]$OnlyThisPc)
$ErrorActionPreference = "Stop"
$region = "ap-northeast-2"
$instance = $env:DEMO_INSTANCE_ID   # 예: i-xxxxxxxxxxxxxxxxx
$sg = $env:DEMO_SECURITY_GROUP_ID   # 예: sg-xxxxxxxxxxxxxxxxx
$tokenFile = Join-Path $PSScriptRoot "DEMO_ACCESS.local.txt"

aws ec2 start-instances --region $region --instance-ids $instance --output text | Out-Null
if ($LASTEXITCODE -ne 0) { throw "인스턴스를 켜지 못했습니다. AWS 로그인(aws login)을 확인하세요." }
aws ec2 wait instance-running --region $region --instance-ids $instance
if ($LASTEXITCODE -ne 0) { throw "인스턴스가 running 상태가 되지 않았습니다." }
$cidr = if ($OnlyThisPc) { (Invoke-RestMethod -Uri "https://checkip.amazonaws.com").Trim() + "/32" } else { "0.0.0.0/0" }
$rules = aws ec2 describe-security-groups --region $region --group-ids $sg --query "SecurityGroups[0].IpPermissions" --output json
if ($rules -ne "[]") {
  $tmp = New-TemporaryFile
  [IO.File]::WriteAllText($tmp, $rules, (New-Object Text.UTF8Encoding $false))
  aws ec2 revoke-security-group-ingress --region $region --group-id $sg --ip-permissions "file://$tmp" | Out-Null
  Remove-Item $tmp
}
aws ec2 authorize-security-group-ingress --region $region --group-id $sg --ip-permissions "IpProtocol=tcp,FromPort=8080,ToPort=8080,IpRanges=[{CidrIp=$cidr,Description=demo}]" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "방화벽 규칙을 등록하지 못했습니다." }
$public = aws ec2 describe-instances --region $region --instance-ids $instance --query "Reservations[0].Instances[0].PublicIpAddress" --output text
$token = ((Get-Content $tokenFile -Raw) -replace "^$([char]0xFEFF)", '' -replace 'DEMO_TOKEN=', '').Trim()
if ($OnlyThisPc) { Write-Host "허용 IP: $cidr (이 PC만 접속 가능)" } else { Write-Host "허용 IP: 모든 IP (토큰이 있는 주소로만 접속 가능) - 끝나면 stop_demo.ps1로 꼭 끄세요" }
Write-Host "모델 로딩까지 약 1~2분 기다린 뒤 아래 주소를 여세요:"
Write-Host "http://${public}:8080/?k=$token"
