contributors=$(git log --format='%ae' | grep '@iscas.ac.cn' | sort | uniq)

# 初始化总贡献数量
total_contributions=0

# 创建临时文件用于保存贡献者的贡献数量
tmp_file=$(mktemp)

# 遍历每个贡献者，统计其贡献数量
for contributor in $contributors; do
	    # 统计贡献者所有贡献中包含'riscv'、'RISCV'、'risc-v'、'RISC-V'信息的提交总数量
	        commit_count=$(git log --grep="arm\|arm64\|armv7\|armv8\|arm-linux\|armhf\|aarch64\|cortex-a" --author="$contributor" --after="2024-01-01" --before="2024-04-01" --format='%H' -i | wc -l)
		    
		    # 累加贡献者的贡献数量到总贡献数量
		        total_contributions=$((total_contributions + commit_count))
			    
			    # 输出贡献者邮箱、贡献数量到临时文件
			        echo "$contributor:$commit_count" >> $tmp_file
			done

			# 软件所内部排名
			echo "--------------------------软件所内部贡献排名--------------------------"

			# 输出每个贡献者的邮箱、贡献数量、排名
			cat $tmp_file | sort -t: -k2nr | awk -F: '{printf "贡献者邮箱: %-50s, 贡献数量: %-5s, 排名: %s\n", $1, $2, NR}' 

			# 输出总贡献数量
			echo "软件所总贡献数量: $total_contributions"

			# 执行git log命令并将结果保存到$tmp_file文件中
			git log --grep="arm\|arm64\|armv7\|armv8\|arm-linux\|armhf\|aarch64\|cortex-a" --after="2024-01-01" --before="2024-04-01" --format='%ae' -i | sort | uniq -c | sort -nr > $tmp_file

			# 统计全球总贡献数量
			global_total_contributions=$(awk '{sum += $1} END {print sum}' $tmp_file)

			# 添加 iscas@iscas.ac.cn 的贡献数量信息到$tmp_file
			echo "$total_contributions iscas@iscas.ac.cn" >> $tmp_file

			# 提取贡献数量信息，排序并排名
			awk '{print $1,$2}' $tmp_file | awk -v OFS="\t" '{print $1,$2}' | sort -nr > contributors_sorted.txt

			# 全球排名
			echo "----------------------------全球贡献排名------------------------------"

			# 输出每个贡献者的邮箱、贡献数量、排名
			awk -F"\t" '{printf "贡献者邮箱: %-50s, 贡献数量: %-5s, 排名: %s\n", $2, $1, NR}' contributors_sorted.txt

			# 输出全球总贡献数量
			echo "全球总贡献数量: $global_total_contributions"

			# 删除临时文件
			rm $tmp_file
			rm contributors_sorted.txt


