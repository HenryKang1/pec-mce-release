$ErrorActionPreference = "Continue"

$Root = "."
$Log = Join-Path $Root "experiments\results\longbench\_gemma4_mce.log"
$Tasks = @("hotpotqa", "2wikimqa", "musique", "qasper", "multifieldqa_en")
$Variants = @(
  "raw_topk", "raw_topk_extractive", "raw_topk_short15", "raw_topk_concise",
  "raw_topk_b840", "raw_topk_b840_extractive", "raw_topk_b840_short15", "raw_topk_b840_concise",
  "sentence_only", "sentence_only_extractive", "sentence_only_short15", "sentence_only_concise",
  "pec_hop", "pec_hop_extractive", "pec_hop_short15", "pec_hop_concise"
)

Set-Location $Root
"=== Gemma-4-E4B MCE run started $(Get-Date -Format o) ===" | Tee-Object -FilePath $Log -Append

foreach ($Task in $Tasks) {
  $N = if ($Task -eq "multifieldqa_en") { 150 } else { 200 }
  foreach ($Variant in $Variants) {
    $OutFile = Join-Path $Root "experiments\results\longbench\gemma-4-e4b_${Task}_${Variant}_n${N}.json"
    if (Test-Path $OutFile) {
      "[Skip] $Task / $Variant -> exists" | Tee-Object -FilePath $Log -Append
      continue
    }
    ">>> $(Get-Date -Format o) task=$Task variant=$Variant n=$N" | Tee-Object -FilePath $Log -Append
    conda run -n research6 python experiments\scripts\longbench_pipeline.py --task $Task --model gemma-4-e4b --variant $Variant --n-samples $N *>> $Log
    if ($LASTEXITCODE -ne 0) {
      "[Error] task=$Task variant=$Variant exit=$LASTEXITCODE" | Tee-Object -FilePath $Log -Append
    }
  }
}

"=== Gemma-4-E4B MCE run finished $(Get-Date -Format o) ===" | Tee-Object -FilePath $Log -Append
