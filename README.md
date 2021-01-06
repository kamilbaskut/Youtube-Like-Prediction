# Youtube-Like-Prediction
Verilen bir youtube linki üzerinden video sayfasına gidilir. Bu sayfa üzerinden, videoyu yayınlayan kanala oradan da kanal’ın yayınladığı videolara gidilir. Buradaki videoların(son yüklenen 30 video) linkleri çekilir. Bu linkler gezilerek izlenme sayısı, beğeni sayısı, beğenmeme sayısı, yorum sayısı ve yayınladığı tarih bilgileri çekilir. Daha sonra bu bilgiler transform edilerek dataframe oluşturulur. Oluşturulan bu dataframe regression yöntemleri ile denenerek beğeni sayısı tahmin edilmeye çalışılmıştır.
# Verilerin Öznitelikleri
'numberOfViews', 'dates', 'numberOfDisLikes', 'numberOfComments', 'links', 'numberOfLikes’

Sitenin dinamik verilerini elde etmek için chrome driver kullanıldı. Sisteminin bunu kullanabilmesi için chrome driverın geçerli bir PATH üzerinde olması gerekir
