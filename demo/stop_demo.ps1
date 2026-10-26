# 실험 데모 종료: 8080 포트 허용 규칙을 지우고 GPU 인스턴스를 정지합니다(삭제 아님). 정지 중에는 디스크 비용만 듭니다.
$region = "ap-northeast-2"
$sg = $env:DEMO_SECURITY_GROUP_ID   # 예: sg-xxxxxxxxxxxxxxxxx
$rules = aws ec2 describe-security-groups --region $region --group-ids $sg --query "SecurityGroups[0].IpPermissions" --output json
if ($LASTEXITCODE -eq 0 -and $rules -ne "[]") {
  $tmp = New-TemporaryFile
  [IO.File]::WriteAllText($tmp, $rules, (New-Object Text.UTF8Encoding $false))
  aws ec2 revoke-security-group-ingress --region $region --group-id $sg --ip-permissions "file://$tmp" | Out-Null
  Remove-Item $tmp
}
aws ec2 stop-instances --region $region --instance-ids $env:DEMO_INSTANCE_ID --query "StoppingInstances[0].CurrentState.Name" --output text
