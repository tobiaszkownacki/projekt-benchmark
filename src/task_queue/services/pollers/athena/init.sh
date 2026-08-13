#!/bin/bash

while true; do
  python task_queue/services/pollers/athena/athena_poller.py
  sleep 60
done