import os
MYSQL={'host':os.getenv('MYSQL_HOST','127.0.0.1'),'port':int(os.getenv('MYSQL_PORT','3306')),'user':os.getenv('MYSQL_USER','admin'),'password':os.getenv('MYSQL_PASSWORD','root'),'db':os.getenv('MYSQL_DB','trading'),'charset':'utf8mb4'}
