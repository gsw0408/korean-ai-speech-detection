# 고친 index.html을 GPU 데모 서버에 올리고 데모 서비스를 다시 시작합니다.
# start_demo.ps1로 서버를 켠 뒤에 실행하세요. 서비스가 다시 뜰 때까지(모델 로딩) 1~2분 걸립니다.
$ErrorActionPreference = "Stop"
$region = "ap-northeast-2"
$instance = $env:DEMO_INSTANCE_ID   # 예: i-xxxxxxxxxxxxxxxxx
$page = Join-Path $PSScriptRoot "index.html"

$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($page))
$hash = (Get-FileHash $page -Algorithm SHA256).Hash.ToLower()
$commands = @(
  "echo $b64 | base64 -d > /opt/exp/demo/index.html.new",
  "test `"`$(sha256sum /opt/exp/demo/index.html.new | cut -d' ' -f1)`" = $hash",
  "mv /opt/exp/demo/index.html.new /opt/exp/demo/index.html",
  "systemctl restart bam-demo",
  "for i in `$(seq 1 60); do curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/ | grep -qv 000 && break; sleep 3; done",
  "systemctl is-active bam-demo"
)
$params = Join-Path ([IO.Path]::GetTempPath()) "push_page_params.json"
[IO.File]::WriteAllText($params, (@{ commands = $commands; executionTimeout = @("600") } | ConvertTo-Json -Compress), (New-Object Text.UTF8Encoding $false))

# SSM 에이전트가 서버 부팅 직후 준비될 때까지 잠시 걸릴 수 있어 몇 번 다시 시도합니다.
$cmdId = $null
foreach ($try in 1..6) {
  $cmdId = aws ssm send-command --region $region --instance-ids $instance --document-name AWS-RunShellScript --parameters "file://$params" --query "Command.CommandId" --output text
  if ($LASTEXITCODE -eq 0 -and $cmdId) { break }
  Write-Host "서버 준비 대기 중... ($try/6)"; Start-Sleep -Seconds 20
}
Remove-Item $params
if (-not $cmdId) { throw "명령을 보내지 못했습니다. start_demo.ps1로 서버를 먼저 켰는지, AWS 로그인이 살아 있는지 확인하세요." }

Write-Host "업로드 후 서비스 재시작 중..."
do {
  Start-Sleep -Seconds 5
  $status = aws ssm get-command-invocation --region $region --command-id $cmdId --instance-id $instance --query "Status" --output text 2>$null
} while ($status -in @("Pending", "InProgress", "Delayed", ""))
if ($status -ne "Success") {
  aws ssm get-command-invocation --region $region --command-id $cmdId --instance-id $instance --query "StandardErrorContent" --output text
  throw "반영 실패: $status"
}
Write-Host "완료: 고친 페이지가 반영됐습니다. start_demo.ps1이 알려 준 주소를 새로고침하세요."
