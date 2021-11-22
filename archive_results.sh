#! /bin/sh

mkdir archive

rsync -a logs/*.log archive
rsync -a checkpoints/* archive
rsync -a tb_logs/* archive

rm -rf logs/*.log
rm -rf checkpoints/*
rm -rf tb_logs/*
