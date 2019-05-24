payload = {}
payload['urls'] = {}

var fun = function(){

	divs = document.getElementsByClassName('product');
	htmls = [];

	for(i=0;i<divs.length;++i){
    	htmls[i] = divs[i].getElementsByClassName('fy-lazy-img')[0].src
	}
	
	htmlSrc = [];
	len = htmls.length;

	final = {};
	k = 0;

	for(i=0;i<len;++i){
		if(htmls[i].indexOf('https') > -1){
			htmlSrc[k++] = htmls[i];
		}
	}
	console.log(htmlSrc);
	for(k=0;k<htmlSrc.length;++k){
		final[k] = [];
		final[k].push(htmlSrc[k]);
		split = htmlSrc[k].split('/');
		join1 = split.slice(0,split.length - 1).join('/');
		img_path = split[split.length - 1];
		img_split = img_path.split('_');
		for(i=2;i<=5;++i){
			img_split[0] = i;
			join2 = img_split.join('_');
			if(join2.indexOf('.jpg') < 0){
				join2 += '.jpg';
			}
			final[k].push(join1 + '/' + join2);
		}
	}
	
	return final;
}

var f = fun();
payload.urls = f;
payload = JSON.stringify(payload);