
COMPARISON_KEYS = "C,H_I,W_I,K,H_O,W_O,T_I,T_O,kH,kW,kT,dH,dW,dT,dilationH,dilationW,dilationT,padH,padW,padT,groups".split(",")

DEDX_SKIP = 
	[ 
	{"topology" => "resnet50", "module"=>"conv1"},
	{"topology" => "densenet169", "module"=>"conv1"},
	{"topology" => "inceptionv3", "module"=>"Conv2d_1a_3x3.conv"},
	{"topology" => "xception", "module"=>"entry.m0.0"},
	{"topology" => "vgg19", "module"=>"features.0"}
	]

DEDX_MEMSET_OUTPUT_LAYERS = 
	[ 
	{"topology" => "resnet50", "module"=>"layer2.0.downsample.0"},
	{"topology" => "resnet50", "module"=>"layer3.0.downsample.0"},
	{"topology" => "resnet50", "module"=>"layer4.0.downsample.0"},
	{"topology" => "xception", "module"=>"entry.res1.0"},
	{"topology" => "xception", "module"=>"entry.res2.0"},
	{"topology" => "xception", "module"=>"entry.res3.0"},
	{"topology" => "xception", "module"=>"exit.res1.0"}
	]

SRAM_SIZE = 19*1024*1024;
	
BATCH = {
	"resnet50" => 2,
}
BATCH.default = 1

def layerMatch(layer, matchList)
	matchList.each do |h|
		match = true
		h.each do |k, v|
			if (v != layer[k])
				match = false
				break
			end
		end
		if match
			return true
		end
	end
	return false
end


	
	
if ARGV.size != 2
	$stderr.puts "usage #{$0} <input_path> <output_file>"
	exit(1)
end

INPUT_PATH = ARGV[0]
OUTPUT_FILE_NAME = ARGV[1]

layers = []

Dir.entries(INPUT_PATH).grep(/\.csv\Z/).each do |file|
	lines = IO.readlines("#{INPUT_PATH}/#{file}")
	lines = lines.collect do |l|
		while (l =~ /\"([^"]*)\"/)
			e = $1.gsub(/,/, '_')
			l = l.sub(/\"([^"]*)\"/, e)
		end
		l.gsub("\s", '_')
	end
	header = lines.shift.split(",").collect {|e| e.strip}
	
	lines.each do |line|
		newLayer = {"topology" => file.sub(/\.csv\Z/, "")}
		params = line.split(",")
		if (params.size != header.size)
			$stderr.puts "ERROR: Header vs. Line size mismatch."
			exit(1)
		end
		params.each_with_index { |p, i| newLayer[header[i]] = p}
		
		addLayer = true
		layers.each do |layer|
			sameLayer = true
			COMPARISON_KEYS.each do |k|
				if layer[k] != newLayer[k]
					sameLayer = false
					break
				end
			end
			if sameLayer
				addLayer = false
				break
			end
		end
		
		if addLayer
			layers.push newLayer
		end
	end
end	

File.open(OUTPUT_FILE_NAME, "w") do |of|

	of.puts "global:pole=north"
	of.puts "global:sramBase=0x7ff00a0000"
	of.puts "global:sramSize=0x1380000"
	of.puts "global:hbmBase=0x20000000"
	of.puts "global:multiplTests=1"
	of.puts "global:fp=1"
	of.puts "global:shuffle=0"
	of.puts "global:programInSram=random"
	of.puts "global:smBase=0x7ffc4f0000"

	layers.each do |layer|
		
		xSize = layer['C'].to_i * layer['W_I'].to_i * layer['H_I'].to_i * layer['T_I'].to_i * BATCH[layer['topology']].to_i
		ySize = layer['K'].to_i * layer['W_O'].to_i * layer['H_O'].to_i * layer['T_O'].to_i * BATCH[layer['topology']].to_i
		wSize = layer['C'].to_i * layer['K'].to_i * layer['kW'].to_i * layer['kH'].to_i * layer['kT'].to_i
		
		largeLayer = ((xSize+ySize+wSize)*4 > SRAM_SIZE)
		skipDedx = layerMatch(layer, DEDX_SKIP)
		memsetDedx = layerMatch(layer, DEDX_MEMSET_OUTPUT_LAYERS)
				
		scaledDedxInput = (layer['dW'].to_i > 1) ||	(layer['dH'].to_i > 1) || (layer['dT'].to_i > 1)
		
		of.puts "testName=#{layer['topology']}_layer_#{layer['num']}_#{layer['module']}"
		of.puts "operation=fwd, #{skipDedx ? '' : 'dedx,' }dedw"
		of.puts "reluEn=random"
		of.puts "lowerEn=random"
		of.puts "sbReuse=random"
		of.puts "adjustFloatShapes=0"
		of.puts "xSizes=#{layer['C']}, #{layer['W_I']}, #{layer['H_I']}, #{layer['T_I']}, #{BATCH[layer['topology']]}"
		of.puts "ySizes=#{layer['K']}, #{layer['W_O']}, #{layer['H_O']}, #{layer['T_O']}, #{BATCH[layer['topology']]}"
		of.puts "wSizes=#{layer['K']}, #{layer['C']}, #{layer['kW']}, #{layer['kH']}, #{layer['kT']}"
		of.puts "dilation=#{layer['dilationW']}, #{layer['dilationH']}, #{layer['dilationT']}"
		of.puts "strides=#{layer['dW']}, #{layer['dH']}, #{layer['dT']}"
		of.puts "padding=#{layer['padW']}, #{layer['padH']}, #{layer['padT']}"
		of.puts "inType=#{largeLayer ? 'bfloat' : 'random'}"
		of.puts "outType=#{largeLayer ? 'bfloat' : 'random'}"
		of.puts "paddingValue=77"
		of.puts "convPattern=random"
		of.puts "dedwPattern=random"
		of.puts "geometry=random"
		of.puts "roundingMode=random"
		of.puts "xInHbm=random_in_dedx"
		of.puts "yInHbm=random_in_fwd"
		of.puts "wInHbm=random_in_dedw"
		of.puts "memsetOutput=#{memsetDedx ? 'dedx' : 0}"
		of.puts "sramStreamingBudget=-1"
		of.puts "xMinVal=-30"
		of.puts "xMaxVal=50"
		of.puts "yMinVal=-1000"
		of.puts "yMaxVal=2000"
		of.puts "wMinVal=-0.3"
		of.puts "wMaxVal=0.2"
		of.puts "skipRef=0"
		of.puts "scaledRandomValues=#{scaledDedxInput ? 'dedx' : 0}"
		of.puts
	end
end

