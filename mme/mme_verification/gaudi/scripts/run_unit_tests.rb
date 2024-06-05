curr = Dir.pwd
Dir.chdir(ARGV[0])
path = Dir.pwd
descNumN = Dir.entries(Dir.pwd).grep(/master/).grep(/local/).grep(/north/).grep(/addr/).size
descNumS = Dir.entries(Dir.pwd).grep(/master/).grep(/local/).grep(/south/).grep(/addr/).size
Dir.chdir(curr)

env = ENV['ENV_NAME']
echo = (ARGV.size > 1) && (ARGV[1] == "ECHO=1")

agus = [
	["master", "local", "north", descNumN],
	["master", "out", "north", descNumN],
	["slave", "local", "north", descNumN],
	["slave", "shared", "north", descNumN],
	["slave", "out", "north", descNumN],
	["master", "local", "south", descNumS],
	["master", "out", "south", descNumS],
	["slave", "local", "south", descNumS],
	["slave", "shared", "south", descNumS],
	["slave", "out", "south", descNumS]]
 
wdog = [descNumN * 10000, 1000000].max

cmds = [
"#{env}/cad/simulation/runsim -b brain_te -p mme_verif/ -t command_set_brain_te_test -vpd --rf '+BRAIN_TE_DATABASE_PATH=#{path}/ +BRAIN_TE_TYPE=local +BRAIN_TE_START_DESCR=0 +BRAIN_TE_NUM_DESCR=#{descNumN}' -wdog #{wdog}",
"#{env}/cad/simulation/runsim -b brain_te -p mme_verif/ -t command_set_brain_te_test -vpd --rf '+BRAIN_TE_DATABASE_PATH=#{path}/ +BRAIN_TE_TYPE=shared +BRAIN_TE_START_DESCR=0 +BRAIN_TE_NUM_DESCR=#{descNumN}' -wdog #{wdog}",
"#{env}/cad/simulation/runsim -b brain_acc -p mme_verif/ -t command_set_brain_acc_test -vpd --rf '+BRAIN_ACC_DATABASE_PATH=#{path}/ +BRAIN_ACC_NUM_DESCR=#{descNumN} +BRAIN_ACC_START_DESCR=0' -wdog #{wdog}"
]

agus.each do |master, type, pole, descs|
	cmd = "#{env}/cad/simulation/runsim -b agu -p mme_verif/ -t command_set_agu_test -vpd --rf '+AGU_DATABASE_PATH=#{path}/ +AGU_TYPE=#{type} +AGU_MME_POLE=#{pole} +AGU_MME_MST_SLV=#{master} +AGU_START_DESCR=0 +AGU_NUM_DESCR=#{descs}' #{type=="out" ? "-tb AGU_OUT_TB" : ""}"
    cmds.push cmd
end

i=1
cmds.each do |cmd|
    puts "Start test \##{i}/#{cmds.size}..."
    puts cmd
    ret = echo || system(cmd)
    if (ret)
       puts "Test \##{i} PASSED."
    else
       puts "Test \##{i} FAILED."
       exit(1)
    end
    i+=1
end

