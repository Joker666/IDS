| Feature No. | Feature Name    | Description                                                                           |
|-------------|----------------|----------------------------------------------------------------------------------------|
| 1           | duration      | Length of the connection (in seconds).                                                  |
| 2           | protocol_type | Type of network protocol (e.g., tcp, udp, icmp).                                       |
| 3           | service       | The service requested during the connection (e.g., http, ftp, smtp).                   |
| 4           | flag          | Status of the connection (e.g., SF for successful, REJ for rejected).                  |
| 5           | src_bytes     | Bytes sent from source to destination.                                                 |
| 6           | dst_bytes     | Bytes sent from destination to source.                                                 |
| 7           | land          | Binary (1 or 0); 1 means the source and destination are the same (attack attempt).     |
| 8           | wrong_fragment| Number of wrong packet fragments.                                                      |
| 9           | urgent        | Number of urgent packets in the connection.                                            |
| 10          | hot           | Number of "hot" indicators (e.g., commands used in FTP, accesses to system files).     |
| 11          | num_failed_logins | Number of failed login attempts.                                                    |
| 12          | logged_in     | Binary (1 or 0); 1 means the user successfully logged in.                               |
| 13          | num_compromised| Number of compromised conditions (e.g., file modifications).                           |
| 14          | root_shell    | Binary (1 or 0); 1 means root shell access was obtained.                                |
| 15          | su_attempted  | Binary (1 or 0); 1 means an attempt to switch to superuser (root).                     |
| 16          | num_root      | Number of root accesses.                                                                |
| 17          | num_file_creations| Number of file creation operations.                                                 |
| 18          | num_shells    | Number of shell prompts opened.                                                         |
| 19          | num_access_files| Number of accesses to control files (e.g., /etc/passwd).                              |
| 20          | num_outbound_cmds| Number of outbound commands in FTP sessions.                                         |
| 21          | is_host_login | Binary (1 or 0); 1 if the login was from a privileged host.                             |
| 22          | is_guest_login| Binary (1 or 0); 1 if the login was as a guest.                                         |
| 23          | count         | Number of connections to the same host in the past 2 seconds.                          |
| 24          | srv_count     | Number of connections to the same service in the past 2 seconds.                       |
| 25          | serror_rate   | Percentage of connections with SYN errors (failed TCP handshakes).                     |
| 26          | srv_serror_rate| Percentage of SYN errors among connections to the same service.                       |
| 27          | rerror_rate   | Percentage of connections with REJ errors (connection refused).                        |
| 28          | srv_rerror_rate| Percentage of REJ errors among connections to the same service.                       |
| 29          | same_srv_rate | Percentage of connections to the same service.                                         |
| 30          | diff_srv_rate | Percentage of connections to different services.                                       |
| 31          | srv_diff_host_rate| Percentage of connections to different hosts.                                       |
| 32          | dst_host_count| Number of connections to the same destination host.                                    |
| 33          | dst_host_srv_count| Number of connections to the same service at the destination.                      |
| 34          | dst_host_same_srv_rate| Percentage of connections to the same service at the destination.              |
| 35          | dst_host_diff_srv_rate| Percentage of connections to different services at the destination.            |
| 36          | dst_host_same_src_port_rate| Percentage of connections from the same source port.                      |
| 37          | dst_host_srv_diff_host_rate| Percentage of connections to different hosts for the same service.        |
| 38          | dst_host_serror_rate| Percentage of SYN errors for the same service at the destination host.          |
| 39          | dst_host_srv_serror_rate| Percentage of SYN errors for the same service at the destination host.      |
| 40          | dst_host_rerror_rate| Percentage of REJ errors at the destination host.                               |
| 41          | dst_host_srv_rerror_rate| Percentage of REJ errors for the same service at the destination host.      |
