import os
import sys
from astropy.io import fits
import numpy as np
import glob
from sklearn.model_selection import train_test_split

#Preprocess our data
def preprocess_data():
	print("Loading Host Galaxies")
	#Host Galaxies
	files_host = np.sort(glob.glob('/scratch/sbenzvi_lab/desi/time-domain/bgs/hosts/*/*coadd.fits'))
	flux_host = []
	for f in files_host:
		h = fits.open(f)
		fl = h[1].data
		flux_host.append(fl)
		h.close()
	fluxes_hosts = np.concatenate(flux_host)
	nonzero_hosts = fluxes_hosts.sum(axis=1)!=0
	fluxes_hosts = fluxes_hosts[nonzero_hosts]
	subspec_hosts = np.median(fluxes_hosts[:,:6000].reshape(-1,400,15),2)
	maxflux = fluxes_hosts.max(axis=-1).reshape(-1,1)
	minflux = fluxes_hosts.min(axis=-1).reshape(-1,1)
	standarized_hosts = (subspec_hosts - minflux)/(maxflux-minflux)
	del minflux, maxflux, flux_host, subspec_hosts, fluxes_hosts

	print("Loading Type 1As")
	#Type 1A
	files = np.sort(glob.glob('/scratch/sbenzvi_lab/desi/time-domain/bgs/sne_ia/*/*coadd.fits'))
	flux = []
	for f in files:
		h = fits.open(f)
		f = h[1].data
		zeros = np.zeros(400)
		flux.append(f)
		h.close()
	fluxes = np.concatenate(flux)
	nonzero = fluxes.sum(axis=1)!=0
	fluxes = fluxes[nonzero]
	subspec = np.median(fluxes[:,:6000].reshape(-1,400,15),2)
	maxflux = fluxes.max(axis=-1).reshape(-1,1)
	minflux = fluxes.min(axis=-1).reshape(-1,1)
	standarized = (subspec - minflux)/(maxflux-minflux)
	del minflux, maxflux, flux, subspec, fluxes

	print("Loading Type 2Ps")
	#Type 2P
	files_iip = np.sort(glob.glob('/scratch/sbenzvi_lab/desi/time-domain/bgs/sne_iip/*/*coadd.fits'))
	flux_iip = []
	for f in files_iip:
		h = fits.open(f)
		f = h[1].data
		flux_iip.append(f)
		h.close()
	fluxes_iip = np.concatenate(flux_iip)
	nonzero_iip = fluxes_iip.sum(axis=1)!=0
	fluxes_iip = fluxes_iip[nonzero_iip]
	subspec_iip = np.median(fluxes_iip[:,:6000].reshape(-1,400,15),2)
	maxflux = fluxes_iip.max(axis=-1).reshape(-1,1)
	minflux = fluxes_iip.min(axis=-1).reshape(-1,1)
	standarized_iip = (subspec_iip - minflux)/(maxflux-minflux)
	del minflux, maxflux, flux_iip, subspec_iip, fluxes_iip	
	
	print("Loading True Values")
	#Loading Truth Values
	#sne_iip
	info_files = ['/'.join(f.split('/')[:-1])+ '/{}truth.fits'.format(f.split('/')[-1][:-10]) for f in files_iip]
	#info_files = np.sort(glob.glob('/scratch/sbenzvi_lab/desi/time-domain/bgs/sne_iip/*/*truth.fits'))
	rfr_ = []
	truez_ = []
	rmags_ =[]
	epochs_=[]
	for f in info_files:
		h = fits.open(f)
		r= h[3].data['SNE_FLUXRATIO']
		rfr_.append(r)
		z = h[3].data['TRUEZ']
		truez_.append(z)
		m = 22.5 - 2.5*np.log10(h[3].data['FLUX_R'])
		rmags_.append(m)
		e=h[3].data['SNE_EPOCH']
		epochs_.append(e)
		h.close()
	rfr_iip = np.concatenate(rfr_)[nonzero_iip]
	truez_iip = np.concatenate(truez_)[nonzero_iip]
	rmags_iip = np.concatenate(rmags_)[nonzero_iip]
	epochs_iip = np.concatenate(epochs_).astype(int)[nonzero_iip]
	del rfr_, truez_, rmags_, epochs_

	#sne_ia
	info_files = np.sort(glob.glob('/scratch/sbenzvi_lab/desi/time-domain/bgs/sne_ia/*/*truth.fits'))
	rfr_ = []
	truez_ = []
	rmags_ =[]
	epochs_=[]
	for f in info_files:
		h = fits.open(f)
		r= h[3].data['SNE_FLUXRATIO']
		rfr_.append(r)
		z = h[3].data['TRUEZ']
		truez_.append(z)
		m = 22.5 - 2.5*np.log10(h[3].data['FLUX_R'])
		rmags_.append(m)
		e=h[3].data['SNE_EPOCH']
		epochs_.append(e)
		h.close()
	rfr_ia = np.concatenate(rfr_)[nonzero]
	truez_ia = np.concatenate(truez_)[nonzero]
	rmags_ia = np.concatenate(rmags_)[nonzero]
	epochs_ia = np.concatenate(epochs_).astype(int)[nonzero]
	del rfr_, truez_, rmags_, epochs_

	#Hosts
	info_files = info_files = ['/'.join(f.split('/')[:-1])+ '/{}truth.fits'.format(f.split('/')[-1][:-10]) for f in files_host]
	rfr_ = []
	truez_ = []
	rmags_ =[]
	epochs_=[]
	for f in info_files:
		h = fits.open(f)
		z = h[3].data['TRUEZ']
		truez_.append(z)
		m = 22.5 - 2.5*np.log10(h[3].data['FLUX_R'])
		rmags_.append(m)
		h.close()
	truez_hosts = np.concatenate(truez_)[nonzero_hosts]
	rmags_hosts = np.concatenate(rmags_)[nonzero_hosts]
	del rfr_, truez_, rmags_, epochs_
	
	print("Pre-processing and saving data")
	#Clean up NaN data. This was added to get rid of some warnings that were happening with arithmetic comparisons on NaN
	#rfr_ia_clean = np.array([a if ~np.isnan(a) else 0 for a in rfr_ia])
	#rfr_iip_clean = np.array([a if ~np.isnan(a) else 0 for a in rfr_iip])

	#Get the data we're training on, and 3 labels for them
	x_data = np.concatenate([standarized_hosts[:20000], standarized[:20000], standarized_iip[:20000]]).reshape(-1,400,1)
	y_labels = np.concatenate([np.zeros(20000), np.ones(20000), 1+np.ones(20000)])
	
	x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_labels, test_size=0.1, shuffle=True)	

	x_train_hdu = fits.PrimaryHDU(x_train)
	x_train_hdu.writeto('x_train.fits', overwrite=True)

	y_train_hdu = fits.PrimaryHDU(y_train)
	y_train_hdu.writeto('y_train.fits', overwrite=True)

	x_test_hdu = fits.PrimaryHDU(x_valid)
	x_test_hdu.writeto('x_test.fits', overwrite=True)

	y_test_hdu = fits.PrimaryHDU(y_valid)
	y_test_hdu.writeto('y_test.fits', overwrite=True)
	return;

preprocess_data()
