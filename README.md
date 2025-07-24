To Run the Banking project developed, create a .bat file as below and run the bat file

@echo off

echo Starting Banking App...
start "" cmd /k "cd /d <path-to-reporting-services> && mvn spring-boot:run"

echo Starting Service B...
start "" cmd /k "cd /d <path-to-security-operations> && mvn spring-boot:run"

echo Starting Service C...
start "" cmd /k "cd /d <path-to-core-bank-operations> && mvn spring-boot:run"

echo Starting Frontend...
start "" cmd /k "cd /d <path-to-banking-app> && npm start"
