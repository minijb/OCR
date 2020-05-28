from socket import *

tcp_socket=socket(AF_INET,SOCK_STREAM)
tcp_socket.bind(("127.0.0.1",8081))
tcp_socket.listen(128)
print("------1------")
while True:
    new_socket,client_addr=tcp_socket.accept()
    print(client_addr)
    while True:
        file_name=new_socket.recv(1024).decode("utf-8")
        print("------2------")
        if file_name!="exit":
            print("下载的文件名为： %s" % file_name)
            
             ##########################################
            file_content=None
            try:
                f=open(file_name,"rb")
                file_content=f.read()
                f.close()
            except Exception as ret :
                print("打开文件失败")
            if file_content:
                new_socket.send(file_content)
            else:
                print("文件为空")
            
            ##########################################
        else:
            print("服务结束")
            break 
    new_socket.close()
tcp_socket.close()
