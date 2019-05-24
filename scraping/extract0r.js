var fs = require('fs');


fs.readFile('./test_buffer.txt','utf-8',function(err,data){
	if(!err && data){
		var obj = JSON.parse(data);
		var str = '';
		var urls = obj.urls;
		for(var idx in urls){
			str += idx;
			str += ',';

			for(i=0;i<5;++i){
				str += urls[idx][i];
				str += ',';
			}
			str += 'hook&look';
			str += '\n';
		}

		fs.open('./testData/test_data.csv','a+',function(err,fileDescriptor){
			if(!err && fileDescriptor){
				fs.writeFile(fileDescriptor,str,function(err){
					if(!err){
						fs.close(fileDescriptor,function(err){
							if(err){
								console.log('Error closing file!');
							}
						});
					}
					else {
						console.log('Error writing to file!');
					}
				});
			}
			else {
				console.log('Error opening file!');
			}
		});
	}
	else {
		console.log('Error opening file!');
	}
});